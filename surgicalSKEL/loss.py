import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Any, Optional, Tuple

def softmax_helper_dim1(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, 1)

def reduce_loss(loss, reduction):
    reduction_enum = F._Reduction.get_enum(reduction)
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()
    
def weight_reduce_loss(loss,
                       weight = None,
                       reduction = 'mean',
                       avg_factor = None):
    if weight is not None:
        loss = loss * weight

    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        if reduction == 'mean':
            eps = torch.finfo(torch.float32).eps
            loss = loss.sum() / (avg_factor + eps)
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss

def dice_loss(pred,
              target,
              weight=None,
              eps=1e-3,
              reduction='mean',
              naive_dice=False,
              avg_factor=None):
    
    input = pred.flatten(1)
    target = target.flatten(1).float()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input = input.to(device)
    target = target.to(device)

    a = torch.sum(input * target, 1)
    if naive_dice:
        b = torch.sum(input, 1)
        c = torch.sum(target, 1)
        d = (2 * a + eps) / (b + c + eps)
    else:
        b = torch.sum(input * input, 1) + eps
        c = torch.sum(target * target, 1) + eps
        d = (2 * a) / (b + c)
   
    
    loss = 1 - d
    if weight is not None:
        assert weight.ndim == loss.ndim
        assert len(weight) == len(pred)
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss

def print_if_rank0(*args):
    from torch import distributed
    if distributed.get_rank() == 0:
        print(*args)

class AllGatherGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, tensor: torch.Tensor, group: Optional["torch.distributed.ProcessGroup"] = None) -> torch.Tensor:
        ctx.group = group
        gathered_tensor = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(gathered_tensor, tensor, group=group)
        gathered_tensor = torch.stack(gathered_tensor, dim=0)
        return gathered_tensor

    @staticmethod
    def backward(ctx: Any, *grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        grad_output = torch.cat(grad_output)
        torch.distributed.all_reduce(grad_output, op=torch.distributed.ReduceOp.SUM, async_op=False, group=ctx.group)
        return grad_output[torch.distributed.get_rank()], None


# class RobustCrossEntropyLoss(nn.Module):
#     """
#     A compatibility layer that uses BCELoss.
#     This loss expects:
#       - input: probabilities in [0,1] with shape (B, H, W)
#       - target: binary mask with the same shape.
#     """
#     def __init__(self, **kwargs):
#         super(RobustCrossEntropyLoss, self).__init__()
#         self.ignore_index = kwargs.get('ignore_index', None)
#         self.bce = nn.BCELoss(**kwargs)
    
#     def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         # Both input and target are assumed to have shape (B, H, W)
#         return self.bce(input, target.float())

class SkeletonRecallLoss(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False, do_bg: bool = True, smooth: float = 1.,
                 ddp: bool = True):
        """
        A simplified loss for skeleton recall.
        Note: It does not work with background (do_bg must be False).
        Assumes inputs are of shape (B, H, W).
        """
        super(SkeletonRecallLoss, self).__init__()
        if do_bg:
            raise RuntimeError("skeleton recall does not work with background")
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ddp = ddp

    def forward(self, x: torch.Tensor, y: torch.Tensor, loss_mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        # if self.apply_nonlin is not None:
        #     x = self.apply_nonlin(x)
        axes = (1, 2)

        with torch.no_grad():
            if x.shape != y.shape:
                y = y.view(x.shape)
            y_onehot = y  
            sum_gt = y_onehot.sum(axes) if loss_mask is None else (y_onehot * loss_mask).sum(axes)
        
        inter_rec = (x * y_onehot).sum(axes) if loss_mask is None else (x * y_onehot * loss_mask).sum(axes)
        
        if self.ddp and self.batch_dice:
            inter_rec = AllGatherGrad.apply(inter_rec).sum(0)
            sum_gt = AllGatherGrad.apply(sum_gt).sum(0)
        
        if self.batch_dice:
            inter_rec = inter_rec.sum(0)
            sum_gt = sum_gt.sum(0)
        
        rec = (inter_rec + self.smooth) / (torch.clamp(sum_gt + self.smooth, min=1e-8))
        rec = rec.mean()
        return -rec

class AdaptiveSkeletonLoss(nn.Module):
    def __init__(self, smooth=1.0, apply_nonlin=None):
        super(AdaptiveSkeletonLoss, self).__init__()
        self.smooth = smooth
        self.apply_nonlin = apply_nonlin
        
        self.register_buffer('neighbor_kernel', self._create_neighbor_kernel())
    
    def _create_neighbor_kernel(self):
        kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32)
        kernel[0, 0, 1, 1] = 0  # Center pixel
        return kernel
    
    def _analyze_skeleton(self, skeleton):
        neighbors = F.conv2d(skeleton, self.neighbor_kernel, padding=1)
        
        endpoints = (neighbors == 1) & (skeleton > 0.5)  # Exactly 1 neighbor
        midpoints = (neighbors == 2) & (skeleton > 0.5)  # Exactly 2 neighbors
        junctions = (neighbors > 2) & (skeleton > 0.5)   # More than 2 neighbors
        
        return endpoints, midpoints, junctions
    
    def _structural_similarity(self, pred_skel, gt_skel):
        pred_endpoints, pred_midpoints, pred_junctions = self._analyze_skeleton(pred_skel)
        gt_endpoints, gt_midpoints, gt_junctions = self._analyze_skeleton(gt_skel)
        
        def calculate_iou(pred, gt):
            intersection = torch.sum(pred * gt, dim=(2, 3))
            union = torch.sum(pred, dim=(2, 3)) + torch.sum(gt, dim=(2, 3)) - intersection
            return (intersection + self.smooth) / (union + self.smooth)
        
        endpoint_iou = calculate_iou(pred_endpoints, gt_endpoints)
        midpoint_iou = calculate_iou(pred_midpoints, gt_midpoints)
        junction_iou = calculate_iou(pred_junctions, gt_junctions)
        
        gt_endpoint_count = torch.sum(gt_endpoints, dim=(2, 3))
        gt_junction_count = torch.sum(gt_junctions, dim=(2, 3))
        gt_midpoint_count = torch.sum(gt_midpoints, dim=(2, 3))
        gt_total_count = gt_endpoint_count + gt_junction_count + gt_midpoint_count + self.smooth
        
        endpoint_weight = gt_endpoint_count / gt_total_count
        junction_weight = gt_junction_count / gt_total_count
        midpoint_weight = gt_midpoint_count / gt_total_count
        
        structural_loss = 1 - (
            endpoint_weight * endpoint_iou + 
            junction_weight * junction_iou + 
            midpoint_weight * midpoint_iou
        )
        
        return structural_loss.mean()
    
    def _medial_axis_distance(self, pred_skel, gt_skel):
        def distance_approximation(target, reference):
            dilated = reference.clone()
            max_dist = 10
            
            distances = torch.zeros_like(target)
            found_mask = torch.zeros_like(target, dtype=torch.bool)
            
            for d in range(1, max_dist + 1):
                dilated = F.max_pool2d(dilated, kernel_size=3, stride=1, padding=1)
                
                new_points = (dilated > 0.5) & (target > 0.5) & (~found_mask)
                distances[new_points] = d
                found_mask = found_mask | new_points
                
                if torch.all(found_mask | (target <= 0.5)):
                    break
            
            # Set max distance for unfound points
            target_mask = (target > 0.5)
            unfound_points = target_mask & (~found_mask)
            distances[unfound_points] = max_dist
            
            total_points = torch.sum(target > 0.5, dim=(2, 3))
            total_dist = torch.sum(distances, dim=(2, 3))
            
            avg_dist = total_dist / (total_points + self.smooth)
            return avg_dist
        
        pred_to_gt = distance_approximation(pred_skel, gt_skel)
        gt_to_pred = distance_approximation(gt_skel, pred_skel)
        
        mean_distance = (pred_to_gt + gt_to_pred) / 2
        
        # Normalize to [0, 1] range
        normalized_dist = mean_distance / 10  # max_dist from above
        
        return normalized_dist.mean()
    
    def forward(self, pred_skel, gt_skel):
        if self.apply_nonlin is not None:
            pred_skel = self.apply_nonlin(pred_skel)
        
        # Ensure proper shape
        if pred_skel.ndim == 3:
            pred_skel = pred_skel.unsqueeze(1)
        if gt_skel.ndim == 3:
            gt_skel = gt_skel.unsqueeze(1)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        pred_skel = pred_skel.to(device)
        gt_skel = gt_skel.to(device)

        intersection = torch.sum(pred_skel * gt_skel, dim=(2, 3))
        pred_sum = torch.sum(pred_skel, dim=(2, 3))
        gt_sum = torch.sum(gt_skel, dim=(2, 3))
        
        dice = (2 * intersection + self.smooth) / (pred_sum + gt_sum + self.smooth)
        dice_loss = 1 - dice.mean()
        
        structural_loss = self._structural_similarity(pred_skel, gt_skel)
        medial_loss = self._medial_axis_distance(pred_skel, gt_skel)
        
        avg_magnitude = (dice_loss + structural_loss + medial_loss) / 3
        balanced_dice = dice_loss / (dice_loss + self.smooth) * avg_magnitude
        balanced_structural = structural_loss / (structural_loss + self.smooth) * avg_magnitude
        balanced_medial = medial_loss / (medial_loss + self.smooth) * avg_magnitude
        
        total_loss = balanced_dice + balanced_structural + balanced_medial
        
        return total_loss


class clDiceLoss(nn.Module):
    def __init__(self, iter_=3, smooth = 1., exclude_background=False):
        super(clDiceLoss, self).__init__()
        self.iter = iter_
        self.smooth = smooth
        self.exclude_background = exclude_background

    def forward(self, mask_true, mask_pred, skel_true, skel_pred):
        tprec = (torch.sum(torch.multiply(skel_pred, mask_true))+self.smooth)/(torch.sum(skel_pred)+self.smooth)    
        tsens = (torch.sum(torch.multiply(skel_true, mask_pred))+self.smooth)/(torch.sum(skel_true)+self.smooth)    
        cl_dice = 1.0 - 2.0*(tprec*tsens)/(tprec+tsens)
        return cl_dice


class DiceLoss(nn.Module):
    def __init__(self,
                 use_sigmoid=True,
                 activate=True,
                 reduction='mean',
                 naive_dice=False,
                 loss_weight=1.0,
                 eps=1e-3):
        super(DiceLoss, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.reduction = reduction
        self.naive_dice = naive_dice
        self.loss_weight = loss_weight
        self.eps = eps
        self.activate = activate
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor, weight=None, reduction_override=None, avg_factor=None) -> torch.Tensor:
        reduction = reduction_override if reduction_override else self.reduction
        

        if self.activate:
            if self.use_sigmoid:
                pred = pred.sigmoid()
            else:
                raise NotImplementedError("Only sigmoid activation is implemented.")

        loss = self.loss_weight * dice_loss(pred, target, weight, eps=self.eps,
                                             reduction=reduction,
                                             naive_dice=self.naive_dice,
                                             avg_factor=avg_factor)
        return loss

# class MemoryEfficientSoftDiceLoss(nn.Module):
#     def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False, do_bg: bool = True, smooth: float = 1.,
#                  ddp: bool = True):
#         """
#         Saves memory compared to standard implementations.
#         Assumes input shape (B, H, W).
#         """
#         super(MemoryEfficientSoftDiceLoss, self).__init__()
#         self.do_bg = do_bg
#         self.batch_dice = batch_dice
#         self.apply_nonlin = apply_nonlin
#         self.smooth = smooth
#         self.ddp = ddp

#     def forward(self, x: torch.Tensor, y: torch.Tensor, loss_mask: Optional[torch.Tensor]=None) -> torch.Tensor:
#         if self.apply_nonlin is not None:
#             x = self.apply_nonlin(x)
#         axes = (1, 2)  # spatial dimensions

#         with torch.no_grad():
#             if x.shape != y.shape:
#                 y = y.view(x.shape)
#             y_onehot = y  # binary target
#             sum_gt = y_onehot.sum(axes) if loss_mask is None else (y_onehot * loss_mask).sum(axes)
        
#         intersect = (x * y_onehot).sum(axes) if loss_mask is None else (x * y_onehot * loss_mask).sum(axes)
#         sum_pred = x.sum(axes) if loss_mask is None else (x * loss_mask).sum(axes)
        
#         if self.batch_dice:
#             if self.ddp:
#                 intersect = AllGatherGrad.apply(intersect).sum(0)
#                 sum_pred = AllGatherGrad.apply(sum_pred).sum(0)
#                 sum_gt = AllGatherGrad.apply(sum_gt).sum(0)
#             else:
#                 intersect = intersect.sum(0)
#                 sum_pred = sum_pred.sum(0)
#                 sum_gt = sum_gt.sum(0)
        
#         dc = (2 * intersect + self.smooth) / (torch.clamp(sum_gt + sum_pred + self.smooth, min=1e-8))
#         dc = dc.mean()
#         return -dc

class CombinedLoss(nn.Module):
    def __init__(self, weight_cldice=0.0, weight_dice=1.0, weight_srec=0.0, 
                 soft_skelrec_kwargs: Optional[dict] = None):
        """
        Combines three loss components:
          1. clDice loss computed on skeleton predictions,
          2. Dice loss computed on segmentation mask predictions,
          3. Skeleton recall loss computed between the predicted mask and the ground truth skeleton.
        Assumes:
          - mask_pred and mask_gt are of shape (B, H, W)
          - skeleton_pred and skeleton_gt are of shape (B, H, W)
        """
        super(CombinedLoss, self).__init__()
        self.weight_cldice = weight_cldice
        self.weight_dice = weight_dice
        self.weight_srec = weight_srec
        
        self.cldice = clDiceLoss()
        self.dice = DiceLoss()  
        self.srec = SkeletonRecallLoss(apply_nonlin=None, **(soft_skelrec_kwargs or {}))
    
    def forward(self, mask_pred: torch.Tensor, mask_gt: torch.Tensor, 
                skeleton_pred: torch.Tensor, skeleton_gt: torch.Tensor) -> torch.Tensor:
        loss_dice = self.dice(mask_pred, mask_gt)
        # mask_pred_prob = mask_pred.sigmoid()
        # loss_cldice = self.cldice(mask_gt, mask_pred_prob, skeleton_gt, skeleton_pred)
        # loss_srec = self.srec(mask_pred_prob, skeleton_gt)
        # total_loss = (self.weight_cldice * loss_cldice + 
        #               self.weight_dice * loss_dice + 
        #               self.weight_srec * loss_srec)
        total_loss = loss_dice
        return total_loss

class CombinedLoss2(nn.Module):
    def __init__(self, weight_dice=1.0, weight_skeleton=0.3):
        super(CombinedLoss2, self).__init__()
        self.weight_dice = weight_dice
        self.weight_skeleton = weight_skeleton
        
        self.dice = DiceLoss()  
        self.skeleton_loss = AdaptiveSkeletonLoss()
    
    def forward(self, mask_pred, mask_gt, skeleton_pred, skeleton_gt):
        loss_dice = self.dice(mask_pred, mask_gt)
        loss_skeleton = self.skeleton_loss(skeleton_pred, skeleton_gt)
        
        total_loss = self.weight_dice * loss_dice + self.weight_skeleton * loss_skeleton
        
        return total_loss