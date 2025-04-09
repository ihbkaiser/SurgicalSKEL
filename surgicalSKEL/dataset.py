import os
import os.path as osp
import re
import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
import sys
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from segment_anything.modeling.prompt_encoder import PositionEmbeddingRandom

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from segment_anything import sam_model_registry, SamPredictor
def fps_sampling(skel_tensor, num_samples=32):
    skel_np = skel_tensor.squeeze(0).cpu().numpy()  
    points = np.argwhere(skel_np > 0) 
    if points.shape[0] == 0:
        return torch.zeros((num_samples, 2), dtype=torch.float32)
    points_xy = points[:, [1, 0]].astype(np.float32)
    if points_xy.shape[0] < num_samples:
        pad = num_samples - points_xy.shape[0]
        points_xy = np.concatenate([points_xy, np.tile(points_xy[-1:], (pad, 1))], axis=0)
        return torch.from_numpy(points_xy)
    # FPS algorithm
    selected = []
    selected.append(points_xy[0])
    distances = np.full((points_xy.shape[0],), np.inf)
    for _ in range(1, num_samples):
        last_selected = selected[-1]
        dists = np.sum((points_xy - last_selected)**2, axis=1)
        distances = np.minimum(distances, dists)
        index = np.argmax(distances)
        selected.append(points_xy[index])
    selected = np.stack(selected, axis=0)
    return torch.from_numpy(selected)
def compute_class_embedding(sam_feats, mask_tensor):
    feat_h, feat_w, feat_c = sam_feats.shape
    mask_resized = F.interpolate(mask_tensor.unsqueeze(0), size=(feat_h, feat_w), mode='nearest').squeeze(0)
    features_tensor = torch.from_numpy(sam_feats).permute(2, 0, 1).unsqueeze(0).float()
    if mask_resized.sum() == 0:
        return torch.zeros(feat_c)
    masked_features = features_tensor * mask_resized.unsqueeze(0)
    class_embedding = masked_features.sum(dim=(2,3)) / mask_resized.sum()
    return class_embedding.squeeze(0)

class Endovis18Loader(Dataset):
    def __init__(self, data_root_dir="../data/endovis_2018", mode="val", vit_mode="h",
                 sam_predictor=None, transform=None, 
                 precomputed_sam_dir="precomputed_sam_embeddings",
                 precomputed_class_dir="precomputed_class_embeddings",
                 num_landmarks=32):
        self.data_root_dir = data_root_dir
        self.mode = mode   # mode (e.g., "val" or "train")
        self.vit_mode = vit_mode
        if sam_predictor is None:
            raise ValueError("sam_predictor must be provided!")
        self.sam_predictor = sam_predictor
        self.transform = transform if transform is not None else transforms.ToTensor()
        self.image_dir = osp.join(data_root_dir, mode, "images")
        self.mask_dir  = osp.join(data_root_dir, mode, "binary_annotations")
        self.mask_list = []
        for seq in os.listdir(self.mask_dir):
            seq_path = osp.join(self.mask_dir, seq)
            if os.path.isdir(seq_path):
                for mask_file in sorted(os.listdir(seq_path)):
                    if "class" in mask_file and mask_file.lower().endswith(".png"):
                        self.mask_list.append((seq, mask_file))
        self.precomputed_sam_dir = precomputed_sam_dir
        self.precomputed_class_dir = precomputed_class_dir
        self.num_landmarks = num_landmarks
        self.pos_encoder = PositionEmbeddingRandom(num_pos_feats=128)

    def __len__(self):
        return len(self.mask_list)

    def __getitem__(self, index):
        t1 = time.time()
        seq, mask_file = self.mask_list[index]
        mask_path = osp.join(self.mask_dir, seq, mask_file)
        image_id = mask_file.split('_')[0]
        image_path = osp.join(self.image_dir, seq, image_id + ".png")
        skeleton_path = osp.join(self.mask_dir, seq, "skeletons", mask_file)
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask file not found: {mask_path}")
        mask = (mask > 127).astype(np.uint8)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()
        skeleton = cv2.imread(skeleton_path, cv2.IMREAD_GRAYSCALE)
        if skeleton is None:
            raise FileNotFoundError(f"Skeleton file not found: {skeleton_path}")
        skeleton = (skeleton > 127).astype(np.uint8)
        skeleton_tensor = torch.from_numpy(skeleton).unsqueeze(0).float()
        H, W = skeleton_tensor.shape[1], skeleton_tensor.shape[2]
        point_coords = fps_sampling(skeleton_tensor, self.num_landmarks)
        point_coords = point_coords.unsqueeze(0)
        point_embeddings = self.pos_encoder.forward_with_coords(point_coords, (H, W))
        point_embeddings = point_embeddings.squeeze(0)

        # Load precomputed SAM embedding.
        # Build path: precomputed_sam_embeddings/<dataset_num>/<mode>/<seq>/<image_id>.npy
        # Here we assume data_root_dir ends with the dataset folder (e.g., "endovis_data/18")
        dataset_num = osp.basename(self.data_root_dir)
        precomp_sam_path = osp.join(self.precomputed_sam_dir, dataset_num, self.mode, seq, f"{image_id}.npy")
        if not osp.exists(precomp_sam_path):
            raise FileNotFoundError(f"Precomputed SAM embedding not found: {precomp_sam_path}")
        sam_feats = np.load(precomp_sam_path)
        
        # Get class ID from filename.
        match = re.search(r'class(\d+)', mask_file)
        cls_id = int(match.group(1)) if match else -1
        
        precomp_class_path = osp.join(self.precomputed_class_dir, dataset_num, self.mode, seq, f"{image_id}_class{cls_id}.npy")
        if not osp.exists(precomp_class_path):
            raise FileNotFoundError(f"Precomputed class embedding not found: {precomp_class_path}")
        class_embedding = np.load(precomp_class_path)
        
        sam_feats_tensor = torch.from_numpy(sam_feats).float()
        t2 = time.time()
        return sam_feats_tensor, f"{seq}/{mask_file}", cls_id, mask_tensor, torch.from_numpy(class_embedding).float(), skeleton_tensor, point_embeddings

class Endovis17Loader(Dataset):
    def __init__(self, data_root_dir="../data/endovis_2017", mode="train", fold=0, vit_mode="h",
                 sam_predictor=None, transform=None,
                 precomputed_sam_dir="precomputed_sam_embeddings",
                 precomputed_class_dir="precomputed_class_embeddings"):
        self.data_root_dir = data_root_dir
        self.mode = mode
        self.fold = fold
        self.vit_mode = vit_mode
        if sam_predictor is None:
            raise ValueError("sam_predictor must be provided!")
        self.sam_predictor = sam_predictor
        self.transform = transform if transform is not None else transforms.ToTensor()
        all_folds = list(range(1, 9))
        fold_seq = {0: [1, 3], 1: [2, 5], 2: [4, 8], 3: [6, 7]}
        seqs = [x for x in all_folds if x not in fold_seq[fold]] if mode=="train" else fold_seq[fold]
        self.image_dir = osp.join(data_root_dir, mode, "images")
        self.mask_dir = osp.join(data_root_dir, mode, "binary_annotations")
        self.mask_list = []
        for seq in seqs:
            seq_str = f"seq{seq}"
            seq_mask_path = osp.join(self.mask_dir, seq_str)
            if os.path.exists(seq_mask_path):
                for mask_file in sorted(os.listdir(seq_mask_path)):
                    if "class" in mask_file and mask_file.lower().endswith(".png"):
                        self.mask_list.append((seq_str, mask_file))
        if len(self.mask_list) == 0:
            raise ValueError("No mask files loaded. Check your folder structure and parameters.")
        self.precomputed_sam_dir = precomputed_sam_dir
        self.precomputed_class_dir = precomputed_class_dir

    def __len__(self):
        return len(self.mask_list)

    def __getitem__(self, index):
        seq, mask_file = self.mask_list[index]
        mask_path = osp.join(self.mask_dir, seq, mask_file)
        image_id = mask_file.split('_')[0]
        image_path = osp.join(self.image_dir, seq, image_id + ".jpg")
        skeleton_path = osp.join(self.mask_dir, seq, "skeletons", mask_file)
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask file not found: {mask_path}")
        mask = (mask > 127).astype(np.uint8)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()
        skeleton = cv2.imread(skeleton_path, cv2.IMREAD_GRAYSCALE)
        if skeleton is None:
            raise FileNotFoundError(f"Skeleton file not found: {skeleton_path}")
        skeleton = (skeleton > 127).astype(np.uint8)
        skeleton_tensor = torch.from_numpy(skeleton).unsqueeze(0).float()

        # Load precomputed SAM embedding.
        dataset_num = osp.basename(self.data_root_dir)
        precomp_sam_path = osp.join(self.precomputed_sam_dir, dataset_num, self.mode, seq, f"{image_id}.npy")
        if not osp.exists(precomp_sam_path):
            raise FileNotFoundError(f"Precomputed SAM embedding not found: {precomp_sam_path}")
        sam_feats = np.load(precomp_sam_path)
        
        match = re.search(r'class(\d+)', mask_file)
        cls_id = int(match.group(1)) if match else -1
        
        precomp_class_path = osp.join(self.precomputed_class_dir, dataset_num, self.mode, seq, f"{image_id}_class{cls_id}.npy")
        if not osp.exists(precomp_class_path):
            raise FileNotFoundError(f"Precomputed class embedding not found: {precomp_class_path}")
        class_embedding = np.load(precomp_class_path)
        
        sam_feats_tensor = torch.from_numpy(sam_feats).float()
        return sam_feats_tensor, mask_file, cls_id, mask_tensor, torch.from_numpy(class_embedding).float(), skeleton_tensor

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    vit_mode = "h"
    sam_checkpoint = "ckpt/sam_vit_h_4b8939.pth"
    model_type = f"vit_{vit_mode}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device)
    predictor = SamPredictor(sam)
    
    # For Endovis18Loader, using "val" mode (adjust as needed)
    dataset18 = Endovis18Loader(
        data_root_dir="endovis_data/18",
        mode="val",
        vit_mode=vit_mode,
        sam_predictor=predictor,
        precomputed_sam_dir="precomputed_sam_embeddings",
        precomputed_class_dir="precomputed_class_embeddings"
    )
    dataloader18 = DataLoader(dataset18, batch_size=4, shuffle=True, num_workers=0)
    for batch in dataloader18:
        sam_feats_tensor, mask_name, cls_id, mask_tensor, class_embedding, skeleton_tensor = batch
        print("Endovis18:")
        print("SAM features shape:", sam_feats_tensor.shape)
        print("Mask name:", mask_name)
        print("Class ID:", cls_id)
        print("Mask shape:", mask_tensor.shape)
        print("Class embedding shape:", class_embedding.shape)
        print("Skeleton shape:", skeleton_tensor.shape)
        break

    # For Endovis17Loader, using "train" mode (adjust as needed)
    dataset17 = Endovis17Loader(
        data_root_dir="endovis_data/17",
        mode="train",
        fold=0,
        vit_mode=vit_mode,
        sam_predictor=predictor,
        precomputed_sam_dir="precomputed_sam_embeddings",
        precomputed_class_dir="precomputed_class_embeddings"
    )
    dataloader17 = DataLoader(dataset17, batch_size=4, shuffle=True, num_workers=0)
    for batch in dataloader17:
        sam_feats_tensor, mask_name, cls_id, mask_tensor, class_embedding, skeleton_tensor = batch
        print("Endovis17:")
        print("SAM features shape:", sam_feats_tensor.shape)
        print("Mask name:", mask_name)
        print("Class ID:", cls_id)
        print("Mask shape:", mask_tensor.shape)
        print("Class embedding shape:", class_embedding.shape)
        print("Skeleton shape:", skeleton_tensor.shape)
        break
