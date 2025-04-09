import torch.multiprocessing as mp

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    import sys
    sys.path.append(__import__("os").path.abspath(__import__("os").path.join(__import__("os").path.dirname(__file__), '..')))
    import os
    import os.path as osp 
    import argparse
    import random
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    import time

    from dataset import Endovis18Loader, Endovis17Loader
    from segment_anything import sam_model_registry, SamPredictor

    import torch.nn.functional as F
    import re
    def compute_class_embedding(sam_feats, mask_tensor):
        feat_h, feat_w, feat_c = sam_feats.shape
        mask_resized = F.interpolate(mask_tensor.unsqueeze(0), size=(feat_h, feat_w), mode='nearest').squeeze(0)
        features_tensor = torch.from_numpy(sam_feats).permute(2, 0, 1).unsqueeze(0).float()
        if mask_resized.sum() == 0:
            return torch.zeros(feat_c)
        masked_features = features_tensor * mask_resized.unsqueeze(0)
        class_embedding = masked_features.sum(dim=(2,3)) / mask_resized.sum()
        return class_embedding.squeeze(0)

    parser = argparse.ArgumentParser(description="Precompute SAM and class embeddings for Endovis dataset")
    parser.add_argument('--dataset', type=str, default="18", choices=["18", "17"], help='Dataset number: "18" or "17"')
    parser.add_argument('--mode', type=str, default="val", choices=["train", "val"], help='Which split to precompute embeddings for')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size (recommend 1 for precomputation)')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of DataLoader workers')
    args = parser.parse_args()

    dataset_num = args.dataset
    split_mode = args.mode
    data_root_dir = osp.join("endovis_data", dataset_num)
    batch_size = args.batch_size
    num_workers = args.num_workers
    vit_mode = "h"
    seed = 666

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if vit_mode == "h":
        sam_checkpoint = "ckpt/sam_vit_h_4b8939.pth"
    sam_full = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
    sam_full.to(device)
    predictor = SamPredictor(sam_full)

    if dataset_num == "18":
        DatasetClass = Endovis18Loader
    else:
        DatasetClass = Endovis17Loader

    dataset = DatasetClass(
        data_root_dir=osp.join("endovis_data", dataset_num),
        mode=split_mode,
        vit_mode=vit_mode,
        sam_predictor=predictor
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    out_dir_sam = "precomputed_sam_embeddings"
    out_dir_class = "precomputed_class_embeddings"
    os.makedirs(out_dir_sam, exist_ok=True)
    os.makedirs(out_dir_class, exist_ok=True)
    # Create subfolders for dataset and split (train or val)
    out_dir_sam_dataset = osp.join(out_dir_sam, dataset_num, split_mode)
    out_dir_class_dataset = osp.join(out_dir_class, dataset_num, split_mode)
    os.makedirs(out_dir_sam_dataset, exist_ok=True)
    os.makedirs(out_dir_class_dataset, exist_ok=True)

    def parse_mask_info(mask_name):
        if "/" in mask_name:
            parts = mask_name.split("/")
            seq = parts[0]
            base = parts[1]
        else:
            m = re.match(r'(seq\d+)', mask_name, re.IGNORECASE)
            seq = m.group(1) if m else "seq1"
            base = mask_name
        image_id = osp.basename(base).split("_")[0]
        m_class = re.search(r'class(\d+)', mask_name)
        cls_id = int(m_class.group(1)) if m_class else -1
        return seq, image_id, cls_id

    print(f"Precomputing embeddings for {len(dataset)} samples from {dataset_num}/{split_mode} ...")
    start_time = time.time()
    for sample in tqdm(dataloader, desc="Precompute Embeddings"):
        sam_feats_tensor, mask_name, cls_id, mask_tensor, class_embedding, skeleton_tensor = sample
        sam_feats_tensor = sam_feats_tensor.squeeze(0).cpu().numpy()   # shape: [H, W, C]
        class_embedding = class_embedding.squeeze(0).cpu().numpy()       # shape: [embedding_dim]
        seq, image_id, cls_from_name = parse_mask_info(mask_name[0])
        out_seq_sam = osp.join(out_dir_sam_dataset, seq)
        out_seq_class = osp.join(out_dir_class_dataset, seq)
        os.makedirs(out_seq_sam, exist_ok=True)
        os.makedirs(out_seq_class, exist_ok=True)
        out_file_sam = osp.join(out_seq_sam, f"{image_id}.npy")
        np.save(out_file_sam, sam_feats_tensor)
        out_file_class = osp.join(out_seq_class, f"{image_id}_class{cls_id[0]}.npy")
        np.save(out_file_class, class_embedding)
    end_time = time.time()
    print(f"Precomputation for split '{split_mode}' finished in {end_time - start_time:.2f} seconds.")
