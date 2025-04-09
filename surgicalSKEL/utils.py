import numpy as np 
import cv2 
import torch 
import os 
import os.path as osp 
import re 

def create_binary_masks(binary_masks, preds, preds_quality, mask_names, thr):
    preds = preds.cpu()
    preds_quality = preds_quality.cpu()
    
    pred_masks = (preds > thr).int()
    
    for pred_mask, mask_name, pred_quality in zip(pred_masks, mask_names, preds_quality):        
        if "/" in mask_name:
            seq_name = mask_name.split("/")[0]
        else:
            seq_name = "default"
        
        frame_name = osp.basename(mask_name).split("_")[0]
        
        if seq_name not in binary_masks:
            binary_masks[seq_name] = dict()
        
        if frame_name not in binary_masks[seq_name]:
            binary_masks[seq_name][frame_name] = list()
            
        binary_masks[seq_name][frame_name].append({
            "mask_name": mask_name,
            "mask": pred_mask,
            "mask_quality": pred_quality.item()
        })
        
    return binary_masks
        

def create_endovis_masks(binary_masks, H, W):
    
    endovis_masks = dict()
    
    for seq in binary_masks.keys():
        for frame in binary_masks[seq].keys():
            endovis_mask = np.zeros((H, W), dtype=int)
            binary_masks_list = binary_masks[seq][frame]

            binary_masks_list = sorted(binary_masks_list, key=lambda x: x["mask_quality"])
           
            for binary_mask in binary_masks_list:
                mask_name  = binary_mask["mask_name"]
                m = re.search(r'class(\d+)', mask_name)
                predicted_label = int(m.group(1)) if m else -1
                mask = binary_mask["mask"].numpy()

                endovis_mask[mask == 1] = predicted_label
            if seq == "default":
                key = f"{frame}.png"
            else:
                key = f"{seq}/{frame}.png"
            endovis_masks[key] = endovis_mask
    return endovis_masks


def eval_endovis(endovis_masks, gt_endovis_masks):
    endovis_results = dict()
    num_classes = 7
    
    all_im_iou_acc = []
    all_im_iou_acc_challenge = []
    cum_I, cum_U = 0, 0
    class_ious = {c: [] for c in range(1, num_classes+1)}
    for file_name, prediction in endovis_masks.items():
        if file_name not in gt_endovis_masks:
            print(f"Warning: key {file_name} không có trong ground truth!")
            continue
        full_mask = gt_endovis_masks[file_name]
        
        im_iou = []
        im_iou_challenge = []
        target = full_mask.numpy()
        gt_classes = np.unique(target)
        gt_classes.sort()
        gt_classes = gt_classes[gt_classes > 0] 
        if np.sum(prediction) == 0:
            if target.sum() > 0: 
                all_im_iou_acc.append(0)
                all_im_iou_acc_challenge.append(0)
                for class_id in gt_classes:
                    class_ious[class_id].append(0)
            continue

        gt_classes = torch.unique(full_mask)
        for class_id in range(1, num_classes + 1): 
            current_pred = (prediction == class_id).astype(np.float64)
            current_target = (full_mask.numpy() == class_id).astype(np.float64)

            if current_pred.sum() != 0 or current_target.sum() != 0:
                i, u = compute_mask_IU_endovis(current_pred, current_target)     
                im_iou.append(i/u)
                cum_I += i
                cum_U += u
                class_ious[class_id].append(i/u)
                if class_id in gt_classes:
                    im_iou_challenge.append(i/u)
        
        if len(im_iou) > 0:
            all_im_iou_acc.append(np.mean(im_iou))
        if len(im_iou_challenge) > 0:
            all_im_iou_acc_challenge.append(np.mean(im_iou_challenge))
    final_im_iou = cum_I / (cum_U + 1e-15)
    mean_im_iou = np.mean(all_im_iou_acc)
    mean_im_iou_challenge = np.mean(all_im_iou_acc_challenge)

    final_class_im_iou = torch.zeros(9)
    cIoU_per_class = []
    for c in range(1, num_classes + 1):
        final_class_im_iou[c-1] = torch.tensor(class_ious[c]).float().mean()
        cIoU_per_class.append(round((final_class_im_iou[c-1]*100).item(), 3))
        
    mean_class_iou = torch.tensor([torch.tensor(values).float().mean() for c, values in class_ious.items() if len(values) > 0]).mean().item()
    
    endovis_results["challengIoU"] = round(mean_im_iou_challenge*100, 3)
    endovis_results["IoU"] = round(mean_im_iou*100, 3)
    endovis_results["mcIoU"] = round(mean_class_iou*100, 3)
    endovis_results["mIoU"] = round(final_im_iou*100, 3)
    endovis_results["cIoU_per_class"] = cIoU_per_class
    
    return endovis_results


def compute_mask_IU_endovis(masks, target):
    assert target.shape[-2:] == masks.shape[-2:], "Shape của target và masks không khớp."
    temp = masks * target
    intersection = temp.sum()
    union = ((masks + target) - temp).sum()
    return intersection, union


def read_gt_endovis_masks(data_root_dir="endovis_data/18",
                          mode="val", 
                          fold=None):
    gt_endovis_masks = dict()
    if "18" in data_root_dir:
        gt_path = osp.join(data_root_dir, mode, "annotations")
        for seq in os.listdir(gt_path):
            seq_dir = osp.join(gt_path, seq)
            if not osp.isdir(seq_dir): 
                continue
            for mask_name in os.listdir(seq_dir):
                full_mask_name = f"{seq}/{mask_name}"
                mask_path = osp.join(seq_dir, mask_name)
                mask = torch.from_numpy(cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE))
                gt_endovis_masks[full_mask_name] = mask
    elif "17" in data_root_dir:
        gt_path = osp.join(data_root_dir, mode, "binary_annotations")
        for seq in os.listdir(gt_path):
            seq_dir = osp.join(gt_path, seq)
            if not osp.isdir(seq_dir): 
                continue
            for mask_name in os.listdir(seq_dir):
                full_mask_name = f"{seq}/{mask_name}"
                mask_path = osp.join(seq_dir, mask_name)
                mask = torch.from_numpy(cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE))
                gt_endovis_masks[full_mask_name] = mask
    return gt_endovis_masks


def print_log(str_to_print, log_file):
    print(str_to_print)
    with open(log_file, "a") as file:
        file.write(str_to_print + "\n")
