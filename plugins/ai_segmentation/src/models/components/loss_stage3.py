"""Smooth, normalized stage3 objective; logits enter BCE without a prior sigmoid."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class Stage3Loss(nn.Module):
    def __init__(self, dice_weight=0.4, bce_weight=0.4, skeleton_weight=0.2,
                 spine_error_weight=2.0, neck_error_weight=3.0):
        super().__init__()
        weights = (dice_weight, bce_weight, skeleton_weight)
        if any(not math.isfinite(w) or w < 0 for w in weights) or sum(weights) <= 0:
            raise ValueError('Loss weights must be finite, nonnegative, with positive sum')
        if any(not math.isfinite(w) or w < 1 for w in (spine_error_weight, neck_error_weight)):
            raise ValueError('Local error weights must be finite and >= 1')
        self.weights = tuple(w / sum(weights) for w in weights)
        self.spine_error_weight = spine_error_weight
        self.neck_error_weight = neck_error_weight

    def forward(self, logits, target, spine_roi, neck_roi, skeleton, valid, allow_empty=False):
        # Float32 reductions/BCE also under mixed precision; no hard prediction thresholds.
        logits, target = logits.float(), target.float()
        valid = valid.bool()
        if not allow_empty:
            if valid.is_cuda:
                torch._assert_async(valid.any())
            elif not valid.any():
                raise ValueError('Loss requires at least one valid voxel')
        # Dense masked reductions avoid CUDA nonzero/boolean-index synchronization.
        p = logits.sigmoid() * valid
        y = target * valid
        dice = 1 - (2 * (p * y).sum() + 1) / (p.square().sum() + y.square().sum() + 1)
        voxel_weight = torch.ones_like(target)
        voxel_weight = torch.where(spine_roi > 0, self.spine_error_weight, voxel_weight)
        voxel_weight = torch.where(neck_roi > 0,
                                   torch.maximum(voxel_weight, voxel_weight.new_tensor(self.neck_error_weight)),
                                   voxel_weight)
        errors = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
        # Equal contribution of foreground/background when both exist. Each region is
        # normalized by its OWN weight sum; background volume cannot drown out necks.
        terms, present = [], []
        for selected in ((target > 0.5) & valid, (target <= 0.5) & valid):
            w = voxel_weight * selected
            weight_sum = w.sum()
            terms.append((errors * w).sum() / weight_sum.clamp_min(1))
            present.append((weight_sum > 0).float())
        bce = torch.stack(terms).sum() / torch.stack(present).sum().clamp_min(1)
        sk = (skeleton > 0) & valid & (target > 0.5)
        # Empty local skeleton contributes zero, not an undefined division.
        sk_count = sk.sum()
        skeleton_loss = (sk_count > 0).float() - (p * sk).sum() / sk_count.clamp_min(1)
        total = sum(w * v for w, v in zip(self.weights, (dice, bce, skeleton_loss)))
        return total, dict(dice=dice, bce=bce, skeleton=skeleton_loss)
