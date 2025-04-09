import torch.multiprocessing as mp

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    import sys
    sys.path.append("..")
    import os
    import os.path as osp 
    import random 
    import argparse
    import numpy as np 
    import torch 
    from torch.utils.data import DataLoader, Subset
    from tqdm import tqdm

    from dataset import Endovis18Loader, Endovis17Loader
    from segment_anything import sam_model_registry, SamPredictor
    from prototypes import Learnable_Prototypes, Prototype_Prompt_Encoder
    from utils import print_log, create_binary_masks, create_endovis_masks, eval_endovis, read_gt_endovis_masks
    from model import model_forward_function
    from loss import DiceLoss, CombinedLoss  # CombinedLoss handles segmentation + skeleton recall loss
    from pytorch_metric_learning import losses  # for contrastive loss

    # --------------------- Sinkhorn (OT) Loss Functions ---------------------
    def sinkhorn_iterations(cost_matrix, a, b, epsilon=0.1, n_iters=50):
        K = torch.exp(-cost_matrix / epsilon) + 1e-9  # avoid in-place modification
        u = torch.ones_like(a)
        v = torch.ones_like(b)
        for _ in range(n_iters):
            u = a / (K @ v)
            v = b / (K.t() @ u)
        T = torch.diag(u) @ K @ torch.diag(v)
        transport_cost = torch.sum(T * cost_matrix)
        entropy = -torch.sum(T * torch.log(T + 1e-9))
        total_cost = transport_cost - epsilon * entropy
        return total_cost

    def optimal_transport_loss(prototypes, class_embeddings, cls_ids, epsilon=0.1, n_iters=50):
        num_classes, feat_dim = prototypes.shape
        device = prototypes.device
        target_means = []
        for c in range(num_classes):
            mask_c = (cls_ids == c)
            if mask_c.sum() > 0:
                mean_emb = class_embeddings[mask_c].mean(dim=0)
            else:
                mean_emb = torch.zeros(feat_dim, device=device)
            target_means.append(mean_emb)
        target_means = torch.stack(target_means, dim=0)  # shape (C, D)
        prototypes_exp = prototypes.unsqueeze(1)   # (C, 1, D)
        means_exp = target_means.unsqueeze(0)        # (1, C, D)
        cost_matrix = torch.sum((prototypes_exp - means_exp)**2, dim=2)  # (C, C)
        a = torch.full((num_classes,), 1/num_classes, device=device)
        b = torch.full((num_classes,), 1/num_classes, device=device)
        loss_ot = sinkhorn_iterations(cost_matrix, a, b, epsilon=epsilon, n_iters=n_iters)
        return loss_ot
    # ------------------------------------------------------------------------

    # ---------------- Command-Line Arguments ----------------
    print("======> Process Arguments")
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="18", choices=["18", "17"], help='specify dataset')
    parser.add_argument('--fold', type=int, default=0, choices=[0,1,2,3], help='specify fold number for endovis_2017 dataset')
    parser.add_argument('--loss_type', type=str, default="ot", choices=["contrastive", "ot", "both"],
                        help='choose the prototype loss type: contrastive, ot, or both')
    parser.add_argument('--weight_contrastive', type=float, default=1.0, help='weight for contrastive loss')
    parser.add_argument('--weight_ot', type=float, default=1.0, help='weight for optimal transport loss')
    args = parser.parse_args()
    # --------------------------------------------------------

    print("======> Set Parameters for Training")
    dataset_name = args.dataset
    fold = args.fold
    thr = 0
    seed = 666  
    data_root_dir = f"endovis_data/{dataset_name}"
    batch_size = 12
    vit_mode = "h"

    # Outer device variable:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set seed for reproducibility
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

    print("======> Load SAM for Predictor (with image_encoder)")
    if vit_mode == "h":
        sam_checkpoint = "ckpt/sam_vit_h_4b8939.pth"
    sam_full = sam_model_registry["vit_h"](checkpoint=sam_checkpoint).to(device)
    predictor = SamPredictor(sam_full)

    print("======> Load SAM for Training (no image_encoder)")
    sam_train = sam_model_registry["vit_h_no_image_encoder"](checkpoint=sam_checkpoint)
    sam_prompt_encoder, sam_mask_decoder = sam_train
    sam_prompt_encoder = sam_prompt_encoder.to(device)
    sam_mask_decoder = sam_mask_decoder.to(device)

    print("======> Load Dataset-Specific Parameters")
    train_dataset = Endovis18Loader(
        data_root_dir=data_root_dir,
        mode="train", 
        vit_mode=vit_mode,
        sam_predictor=predictor,
        precomputed_sam_dir="precomputed_sam_embeddings",
        precomputed_class_dir="precomputed_class_embeddings"
    )
    val_dataset = Endovis18Loader(
        data_root_dir=data_root_dir,
        mode="val",  
        vit_mode=vit_mode,
        sam_predictor=predictor,
        precomputed_sam_dir="precomputed_sam_embeddings",
        precomputed_class_dir="precomputed_class_embeddings"
    )
    
    # indices_seq1 = []
    # for i, sample in enumerate(full_train_dataset):
    #     mask_name = sample[1]  # e.g., "seq1/mask_file.png"
    #     seq_folder = mask_name.split('/')[0]
    #     seq_number = ''.join(filter(str.isdigit, seq_folder))
    #     if seq_number == "2":  # filtering for sequence 2 (adjust as needed)
    #         indices_seq1.append(i)
    
    # print(f"=== Debug: Found {len(indices_seq1)} samples for sequence 1 out of {len(full_train_dataset)} total samples.")
    # if not indices_seq1:
    #     raise ValueError("No samples found for sequence 1 in the training set.")
    
    gt_endovis_masks = read_gt_endovis_masks(data_root_dir=data_root_dir, mode="val")
    num_epochs = 500
    lr = 0.001
    save_dir = "./work_dirs/endovis_2018_seq1_exp/"

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print("======> Load Prototypes and Prototype-based Prompt Encoder")
    learnable_prototypes_model = Learnable_Prototypes(num_classes=7, feat_dim=256).to(device)
    protoype_prompt_encoder = Prototype_Prompt_Encoder(
        feat_dim=256, 
        hidden_dim_dense=128, 
        hidden_dim_sparse=128, 
        size=64, 
        num_landmarks=32, 
        num_classes=7
    ).to(device)
     
    # with open(sam_checkpoint, "rb") as f:
    #     state_dict = torch.load(f, map_location=device)
    #     sam_pn_embeddings_weight = {
    #         k.split("prompt_encoder.point_embeddings.")[-1]: v 
    #         for k, v in state_dict.items() 
    #         if k.startswith("prompt_encoder.point_embeddings") and ("0" in k or "1" in k)
    #     }
    #     num_tokens = 2 if "18" in dataset_name else 4
    #     sam_pn_embeddings_weight_ckp = {
    #         "0.weight": torch.cat([sam_pn_embeddings_weight['0.weight'] for _ in range(num_tokens)], dim=0),
    #         "1.weight": torch.cat([sam_pn_embeddings_weight['1.weight'] for _ in range(num_tokens)], dim=0)
    #     }
    #     protoype_prompt_encoder.pn_cls_embeddings.load_state_dict(sam_pn_embeddings_weight_ckp)

    for name, param in learnable_prototypes_model.named_parameters():
        param.requires_grad = True
        
    for name, param in protoype_prompt_encoder.named_parameters():
        if "pn_cls_embeddings" in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
                  
    print("======> Define Optimiser and Loss")
    combined_loss_model = CombinedLoss(
        weight_cldice=0.0, 
        weight_dice=1.0, 
        weight_srec=0.1, 
        soft_skelrec_kwargs={'batch_dice': False, 'do_bg': False, 'smooth': 1.0, 'ddp': True}
    ).to(device)
    
    contrastive_loss_model = losses.NTXentLoss(temperature=0.07).to(device)
    
    optimiser = torch.optim.Adam([
        {'params': learnable_prototypes_model.parameters()},
        {'params': protoype_prompt_encoder.parameters()},
        {'params': sam_mask_decoder.parameters()},  
        {'params': sam_prompt_encoder.parameters(), 'lr': 0.0}
    ], lr=lr, weight_decay=0.0001)

    print("======> Set Saving Directories and Logs")
    os.makedirs(save_dir, exist_ok=True)
    log_file = osp.join(save_dir, "log.txt")
    print_log(str(args), log_file)

    print("======> Start Training and Validation")
    best_challenge_iou_val = -100.0

    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        protoype_prompt_encoder.train()
        sam_mask_decoder.train()
        sam_prompt_encoder.train()
        learnable_prototypes_model.train()

        for batch in tqdm(train_dataloader, desc="Training Batches", leave=False):
            sam_feats, mask_names, cls_ids, masks, class_embeddings, skeletons, point_embeddings = batch
            sam_feats = sam_feats.to(device, non_blocking=True)
            cls_ids = cls_ids.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)        
            skeletons = skeletons.to(device, non_blocking=True)   
            class_embeddings = class_embeddings.to(device, non_blocking=True)
            point_embeddings = point_embeddings.to(device, non_blocking=True)
            
            prototypes = learnable_prototypes_model()  
            
            preds, preds_quality, pred_skeleton, landmark_sparse_pred = model_forward_function(
                protoype_prompt_encoder, 
                sam_prompt_encoder, 
                sam_mask_decoder, 
                sam_feats, 
                prototypes, 
                cls_ids
            )
            

            seg_loss = DiceLoss().to(device)(preds, masks.float())
            landmark_loss = torch.mean((landmark_sparse_pred - point_embeddings)**2) * 0.1
            loss_type = args.loss_type  
            if loss_type == "contrastive":
                prototype_loss = contrastive_loss_model(
                    prototypes, 
                    torch.arange(0, prototypes.size(0), device=device),
                    ref_emb=class_embeddings,
                    ref_labels=cls_ids
                )
            elif loss_type == "ot":
                num_classes = prototypes.shape[0]
                feat_dim = prototypes.shape[1]
                target_means = []
                for c in range(num_classes):
                    mask_c = (cls_ids == c)
                    if mask_c.sum() > 0:
                        mean_emb = class_embeddings[mask_c].mean(dim=0)
                    else:
                        mean_emb = torch.zeros(feat_dim, device=device)
                    target_means.append(mean_emb)
                target_means = torch.stack(target_means, dim=0)  # shape (C, D)
                prototypes_exp = prototypes.unsqueeze(1)   # (C, 1, D)
                means_exp = target_means.unsqueeze(0)        # (1, C, D)
                cost_matrix = torch.sum((prototypes_exp - means_exp)**2, dim=2)  # (C, C)
                a = torch.full((num_classes,), 1/num_classes, device=device)
                b = torch.full((num_classes,), 1/num_classes, device=device)
                prototype_loss = optimal_transport_loss(prototypes, class_embeddings, cls_ids, epsilon=0.1, n_iters=50)
            elif loss_type == "both":
                num_classes = prototypes.shape[0]
                feat_dim = prototypes.shape[1]
                target_means = []
                for c in range(num_classes):
                    mask_c = (cls_ids == c)
                    if mask_c.sum() > 0:
                        mean_emb = class_embeddings[mask_c].mean(dim=0)
                    else:
                        mean_emb = torch.zeros(feat_dim, device=device)
                    target_means.append(mean_emb)
                target_means = torch.stack(target_means, dim=0)
                prototypes_exp = prototypes.unsqueeze(1)
                means_exp = target_means.unsqueeze(0)
                cost_matrix = torch.sum((prototypes_exp - means_exp)**2, dim=2)
                a = torch.full((num_classes,), 1/num_classes, device=device)
                b = torch.full((num_classes,), 1/num_classes, device=device)
                ot_loss = sinkhorn_iterations(cost_matrix, a, b, epsilon=0.1, n_iters=50)
                contrastive_loss = contrastive_loss_model(
                    prototypes, 
                    torch.arange(0, prototypes.size(0), device=device),
                    ref_emb=class_embeddings,
                    ref_labels=cls_ids
                )
                # Adjust weighting between the two losses:
                weight_contrastive = args.weight_contrastive
                weight_ot = args.weight_ot
                prototype_loss = weight_contrastive * contrastive_loss + weight_ot * ot_loss
            else:
                raise ValueError("Unknown loss type specified. Choose from 'contrastive', 'ot', or 'both'.")
                
            total_loss = seg_loss + prototype_loss + landmark_loss

            optimiser.zero_grad()
            total_loss.backward()
            optimiser.step()
        
        # Validation
        binary_masks = dict()
        protoype_prompt_encoder.eval()
        sam_mask_decoder.eval()
        sam_prompt_encoder.eval()
        learnable_prototypes_model.eval()

        with torch.no_grad():
            prototypes = learnable_prototypes_model()
            for batch in tqdm(val_dataloader, desc="Validation Batches", leave=False):
                sam_feats, mask_names, cls_ids, _, class_embeddings, _, _ = batch
                sam_feats = sam_feats.to(device, non_blocking=True)
                cls_ids = cls_ids.to(device, non_blocking=True)
                preds, preds_quality, _, _ = model_forward_function(
                    protoype_prompt_encoder, 
                    sam_prompt_encoder, 
                    sam_mask_decoder, 
                    sam_feats, 
                    prototypes, 
                    cls_ids
                )
                binary_masks = create_binary_masks(binary_masks, preds, preds_quality, mask_names, thr)
        
        endovis_masks = create_endovis_masks(binary_masks, 1024, 1280)
        endovis_results = eval_endovis(endovis_masks, gt_endovis_masks)
                
        print_log(f"Validation - Epoch: {epoch}/{num_epochs-1}; IoU_Results: {endovis_results}", log_file)
        
        if endovis_results["challengIoU"] > best_challenge_iou_val:
            best_challenge_iou_val = endovis_results["challengIoU"]
            torch.save({
                'prototype_prompt_encoder_state_dict': protoype_prompt_encoder.state_dict(),
                'sam_decoder_state_dict': sam_mask_decoder.state_dict(),
                'sam_prompt_encoder_state_dict': sam_prompt_encoder.state_dict(),
                'prototypes_state_dict': learnable_prototypes_model.state_dict(),
            }, osp.join(save_dir, 'model_ckp.pth'))
            print_log(f"Best Challenge IoU: {best_challenge_iou_val:.4f} at Epoch {epoch}", log_file)
