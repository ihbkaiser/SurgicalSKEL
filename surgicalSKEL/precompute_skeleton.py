import os
import os.path as osp
import cv2
import numpy as np
from skimage.morphology import skeletonize

def process_skeletons(data_root_dir, mode):
    mask_dir = osp.join(data_root_dir, mode, "binary_annotations")
    for seq in os.listdir(mask_dir):
        seq_path = osp.join(mask_dir, seq)
        if not osp.isdir(seq_path):
            continue
        skeleton_folder = osp.join(seq_path, "skeletons")
        if not osp.exists(skeleton_folder):
            os.makedirs(skeleton_folder)
        for mask_file in os.listdir(seq_path):
            if not mask_file.lower().endswith(".png"):
                continue
            mask_path = osp.join(seq_path, mask_file)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            mask = (mask > 127).astype(np.uint8)
            skel = skeletonize(mask.astype(bool)).astype(np.uint8) * 255
            out_path = osp.join(skeleton_folder, mask_file)
            
            cv2.imwrite(out_path, skel)
            print(f"Saved {out_path}")

if __name__ == "__main__":
    process_skeletons("endovis_data/17", "train")
    process_skeletons("endovis_data/18", "train")
    process_skeletons("endovis_data/18", "val")
