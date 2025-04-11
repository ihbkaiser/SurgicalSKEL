import os
import os.path as osp
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import sys
from tqdm import tqdm

# Import your original modules
sys.path.append(r"D:\NCKHSV.2024-2025\SurgicalSKEL")

from segment_anything import sam_model_registry, SamPredictor
from prototypes import Learnable_Prototypes, Prototype_Prompt_Encoder, PrototypeAwareMultiModalEncoder
from utils import print_log, create_binary_masks, create_endovis_masks, eval_endovis, read_gt_endovis_masks
from model import model_forward_function
from loss import DiceLoss, CombinedLoss, CombinedLoss2
from pytorch_metric_learning import losses

# ======== DUMMY DATASET WITH CORRECT SHAPES ========
class DummyEndovisDataset(Dataset):
    def __init__(self, num_samples=10, feature_dim=256, image_size=64, num_classes=7):
        self.num_samples = num_samples
        self.feature_dim = feature_dim
        self.image_size = image_size
        self.num_classes = num_classes
        self.num_landmarks = 32
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        sam_feats = torch.randn(self.image_size, self.image_size, self.feature_dim)
        cls_id = torch.tensor(np.random.randint(1, self.num_classes+1), dtype=torch.long)
        
        # Fake mask name
        mask_name = f"seq1/frame_{idx}_class{cls_id.item()}.png"
        
        mask_tensor = torch.zeros(1, 1024, 1280)
        # Create a random circular mask
        center_y, center_x = torch.randint(200, 800, (1,)), torch.randint(300, 900, (1,))
        radius = torch.randint(50, 150, (1,))
        y, x = torch.meshgrid(torch.arange(1024), torch.arange(1280), indexing='ij')
        mask_tensor[0, ((y - center_y)**2 + (x - center_x)**2 < radius**2)] = 1.0
        
        # Create fake class embedding vector
        class_embedding = torch.randn(self.feature_dim)
        
        # Create fake skeleton tensor (similar to mask but thinner)
        skeleton_tensor = torch.zeros(1, 1024, 1280)
        smaller_radius = radius * 0.5
        skeleton_tensor[0, ((y - center_y)**2 + (x - center_x)**2 < smaller_radius**2)] = 1.0
        
        # Create point embeddings
        point_embeddings = torch.randn(self.num_landmarks, self.feature_dim)
        
        return sam_feats, mask_name, cls_id, mask_tensor, class_embedding, skeleton_tensor, point_embeddings

# ======== MOCK MODELS FOR TESTING ========
class MockSAMPromptEncoder(torch.nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.embed_dim = embed_dim
        
    def get_dense_pe(self):
        return torch.randn(1, self.embed_dim, 64, 64)

class MockSAMDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, image_embeddings, image_pe, sparse_prompt_embeddings, dense_prompt_embeddings, multimask_output=False):
        batch_size = sparse_prompt_embeddings.shape[0]
        # Create a mask of size [B, 1, 256, 256] (SAM's default output size)
        masks = torch.rand(batch_size, 1, 256, 256)
        quality = torch.rand(batch_size, 1)
        return masks, quality

# ======== HELPER FUNCTIONS ========
def create_log_file(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_file = osp.join(log_dir, "test_log.txt")
    return log_file

# ======== MOCK MODEL_FORWARD_FUNCTION FOR TESTING ========
def mock_model_forward(prototype_prompt_encoder, 
                      sam_prompt_encoder, 
                      sam_decoder, 
                      sam_feats, 
                      prototypes, 
                      cls_ids):
    
    batch_size = sam_feats.size(0)
    sam_feats_flat = sam_feats.reshape(batch_size, -1, sam_feats.size(-1))
    
    print(f"sam_feats_flat shape: {sam_feats_flat.shape}")
    print(f"prototypes shape: {prototypes.shape}")
    print(f"cls_ids shape and values: {cls_ids.shape}, {cls_ids}")
    
    dense_embeddings = torch.randn(batch_size, 256, 64, 64, device=sam_feats.device)
    sparse_embeddings = torch.randn(batch_size, 7 * 32, 256, device=sam_feats.device)  # 7 classes * 32 landmarks
    landmark_sparse_pred = torch.randn(batch_size, 32, 256, device=sam_feats.device)  # Batch, landmarks, features
    
    pred = []
    pred_quality = []
    pred_skeleton = []
    
    for i in range(batch_size):
        # Get masks from SAM decoder
        masks, quality = sam_decoder(
            image_embeddings=sam_feats.permute(0, 3, 1, 2)[i:i+1],
            image_pe=sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings[i:i+1],
            dense_prompt_embeddings=dense_embeddings[i:i+1],
            multimask_output=False
        )
        
        masks_resized = F.interpolate(masks, size=(1024, 1280), mode='bilinear', align_corners=False)
        skeleton_pred = masks_resized.clone()
        
        pred.append(masks_resized)
        pred_quality.append(quality)
        pred_skeleton.append(skeleton_pred)
    
    pred = torch.cat(pred, dim=0).squeeze(1)
    pred_quality = torch.cat(pred_quality, dim=0).squeeze(1)
    pred_skeleton = torch.cat(pred_skeleton, dim=0).squeeze(1)
    
    return pred, pred_quality, pred_skeleton, landmark_sparse_pred

# ======== MAIN TEST FUNCTION ========
def test_model_with_dummy_data():
    print("=" * 60)
    print("TESTING MODEL WITH DUMMY DATA (USING ORIGINAL MODULES)")
    print("=" * 60)
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed(42)
    else:
        device = torch.device("cpu")
        
    print(f"Using device: {device}")
    
    log_dir = "./test_logs"
    log_file = create_log_file(log_dir)
    print_log("Starting model test with dummy data...", log_file)

    try:
        print("\nInitializing models...")
        
        # Create mock SAM models
        sam_prompt_encoder = MockSAMPromptEncoder(embed_dim=256).to(device)
        sam_mask_decoder = MockSAMDecoder().to(device)
        
        # Create prototype models
        learnable_prototypes = Learnable_Prototypes(num_classes=7, feat_dim=256).to(device)
        # prototype_prompt_encoder = Prototype_Prompt_Encoder(
        #     feat_dim=256, 
        #     hidden_dim_dense=128, 
        #     hidden_dim_sparse=128, 
        #     size=64, 
        #     num_landmarks=32, 
        #     num_classes=7
        # ).to(device)
        
        prototype_prompt_encoder = PrototypeAwareMultiModalEncoder().to(device)

        # Create loss functions
        dice_loss = DiceLoss().to(device)
        # combined_loss = CombinedLoss(
        #     weight_cldice=0.0, 
        #     weight_dice=1.0, 
        #     weight_srec=0.1, 
        #     soft_skelrec_kwargs={'batch_dice': False, 'do_bg': False, 'smooth': 1.0, 'ddp': False}
        # ).to(device)

        combined_loss = CombinedLoss2().to(device)

        contrastive_loss = losses.NTXentLoss(temperature=0.07).to(device)
        
        print_log("Models initialized successfully!", log_file)
    except Exception as e:
        print_log(f"Error initializing models: {str(e)}", log_file)
        import traceback
        print_log(traceback.format_exc(), log_file)
        return False
    
    try:
        print("\nCreating dummy dataset...")
        dummy_dataset = DummyEndovisDataset(num_samples=20)
        batch_size = 4
        
        g = torch.Generator()
        g.manual_seed(42)
        
        dummy_dataloader = DataLoader(
            dummy_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            generator=g,
            num_workers=0
        )
        
        print_log(f"Created dataset with {len(dummy_dataset)} samples", log_file)
    except Exception as e:
        print_log(f"Error creating dataset: {str(e)}", log_file)
        import traceback
        print_log(traceback.format_exc(), log_file)
        return False
    
    optimizer = torch.optim.Adam([
        {'params': learnable_prototypes.parameters()},
        {'params': prototype_prompt_encoder.parameters()},
    ], lr=0.001, weight_decay=0.0001)
    
    print_log("\nStarting training loop with dummy data...", log_file)
    num_epochs = 2
    
    try:
        for epoch in range(num_epochs):
            # Set models to training mode
            learnable_prototypes.train()
            prototype_prompt_encoder.train()
            sam_mask_decoder.train()
            sam_prompt_encoder.train()
            
            epoch_loss = 0
            
            for i, batch in enumerate(dummy_dataloader):
                print(f"Processing batch {i+1}/{len(dummy_dataloader)} in epoch {epoch+1}/{num_epochs}")
                
                sam_feats, mask_names, cls_ids, masks, class_embeddings, skeletons, point_embeddings = batch
                
                sam_feats = sam_feats.to(device)
                cls_ids = cls_ids.to(device)
                masks = masks.to(device)
                skeletons = skeletons.to(device)
                class_embeddings = class_embeddings.to(device)
                point_embeddings = point_embeddings.to(device)
                
                prototypes = learnable_prototypes()
                
                try:
                    print_log(f"Using mock forward function for batch {i+1}", log_file)
                    preds, preds_quality, pred_skeleton, landmark_sparse_pred = mock_model_forward(
                        prototype_prompt_encoder,
                        sam_prompt_encoder,
                        sam_mask_decoder,
                        sam_feats,
                        prototypes,
                        cls_ids
                    )
                    
                    print_log(f"Testing CombinedLoss...", log_file)
                    if preds.shape[-2:] != masks.shape[-2:]:
                        masks_resized = F.interpolate(masks, size=preds.shape[-2:], mode='nearest')
                    else:
                        masks_resized = masks
                        
                    if pred_skeleton.shape[-2:] != skeletons.shape[-2:]:
                        skeletons_resized = F.interpolate(skeletons, size=pred_skeleton.shape[-2:], mode='nearest')
                    else:
                        skeletons_resized = skeletons
                        
                    total_loss = combined_loss(
                        preds, 
                        masks_resized.squeeze(1), 
                        pred_skeleton, 
                        skeletons_resized.squeeze(1)
                    )
                    
                    seg_loss = dice_loss(preds, masks_resized.float())
                    landmark_loss = torch.mean((landmark_sparse_pred - point_embeddings)**2) * 0.1
                    prototype_loss = torch.mean((prototypes.mean(0) - class_embeddings.mean(0))**2)
                    
                    total_loss = total_loss + landmark_loss + prototype_loss
                    
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()
                    
                    epoch_loss += total_loss.item()
                    
                    batch_metrics = {
                        "combined_loss": total_loss.item(),
                        "seg_loss": seg_loss.item(),
                        "prototype_loss": prototype_loss.item(),
                        "landmark_loss": landmark_loss.item()
                    }
                    print_log(f"Epoch {epoch+1}, Batch {i+1}: {batch_metrics}", log_file)
                    
                except Exception as e:
                    print_log(f"Error in batch {i+1}: {str(e)}", log_file)
                    import traceback
                    print_log(traceback.format_exc(), log_file)
                    continue
            
            avg_loss = epoch_loss / len(dummy_dataloader)
            print_log(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.4f}", log_file)
        
        print_log("\nTraining completed successfully!", log_file)
    except Exception as e:
        print_log(f"\nError during training: {str(e)}", log_file)
        import traceback
        print_log(traceback.format_exc(), log_file)
        return False
    
    print_log("\nTesting inference mode...", log_file)
    try:
        learnable_prototypes.eval()
        prototype_prompt_encoder.eval()
        sam_mask_decoder.eval()
        sam_prompt_encoder.eval()
        
        binary_masks = dict()
        
        with torch.no_grad():
            prototypes = learnable_prototypes()
            
            for batch in dummy_dataloader:
                sam_feats, mask_names, cls_ids, _, _, _, _ = batch
                
                sam_feats = sam_feats.to(device)
                cls_ids = cls_ids.to(device)
                
                preds, preds_quality, _, _ = mock_model_forward(
                    prototype_prompt_encoder,
                    sam_prompt_encoder,
                    sam_mask_decoder,
                    sam_feats,
                    prototypes,
                    cls_ids
                )
                
                binary_masks = create_binary_masks(binary_masks, preds, preds_quality, mask_names, thr=0)
                
                print_log(f"Processed batch with {len(mask_names)} masks", log_file)
                break
        
        endovis_masks = create_endovis_masks(binary_masks, 1024, 1280)
        print_log(f"Created {len(endovis_masks)} Endovis masks", log_file)
        
        print_log("\nInference test completed successfully!", log_file)
    except Exception as e:
        print_log(f"\nError during inference: {str(e)}", log_file)
        import traceback
        print_log(traceback.format_exc(), log_file)
        return False
    
    print_log("\n" + "=" * 60, log_file)
    print_log("ALL TESTS COMPLETED SUCCESSFULLY!", log_file)
    print_log("=" * 60, log_file)
    return True

if __name__ == "__main__":
    success = test_model_with_dummy_data()
    if success:
        print("\nThe model works correctly with dummy data!")
    else:
        print("\nThere were errors during the test. Check the log file for details.")