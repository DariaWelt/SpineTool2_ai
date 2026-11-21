"""GT-only supervision maps. Radii and spacing are in micrometres (ZYX)."""
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize


def build_local_targets(mask, spine, shaft, scale, spine_radius_um=0.6, neck_radius_um=0.3):
    spacing = np.asarray(scale, dtype=float)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError('scale must contain positive Z,Y,X voxel sizes in micrometres')
    if not (np.isfinite(spine_radius_um) and np.isfinite(neck_radius_um)
            and spine_radius_um > 0 and neck_radius_um > 0):
        raise ValueError('Local supervision radii must be positive and finite')
    mask, spine, shaft = (np.asarray(x, dtype=bool) for x in (mask, spine, shaft))
    if mask.shape != spine.shape or mask.shape != shaft.shape:
        raise ValueError('GT, spine and shaft masks must have identical shapes')
    # If a full neuron is supplied as shaft_mask, subtract manually marked spines.
    spine = spine & mask
    shaft = shaft & mask & ~spine
    empty = np.zeros(mask.shape, dtype=np.uint8)
    if not spine.any():
        return dict(spine_roi=empty.copy(), neck_roi=empty.copy(), local_skeleton=empty.copy())
    near_spine = distance_transform_edt(~spine, sampling=spacing) <= spine_radius_um
    # A neck candidate is a foreground location close to BOTH spine and shaft.
    # This is a weighting region, never a replacement for the binary GT.
    if shaft.any():
        near_shaft = distance_transform_edt(~shaft, sampling=spacing) <= neck_radius_um
        transition = mask & near_spine & near_shaft
        neck_roi = (distance_transform_edt(~transition, sampling=spacing) <= neck_radius_um
                    if transition.any() else empty.astype(bool))
    else:
        neck_roi = empty.astype(bool)
    local = near_spine | neck_roi
    skeleton = skeletonize(mask, method='lee') > 0
    return dict(spine_roi=near_spine.astype(np.uint8), neck_roi=neck_roi.astype(np.uint8),
                local_skeleton=(skeleton & local).astype(np.uint8))
