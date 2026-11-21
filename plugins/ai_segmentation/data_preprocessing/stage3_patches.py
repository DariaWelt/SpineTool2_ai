"""Bounded-memory crops and virtual whole-volume XY rotations."""
import numpy as np
from scipy import special
from scipy.ndimage import affine_transform


def foreground_crop(mask, spacing, spine_radius_um, neck_radius_um, patch_shape):
    """Keep all GT plus full rotated-patch context and local-loss distance halo.

    Axis projections avoid allocating an N_foreground x 3 coordinate array.
    """
    spacing = np.asarray(spacing, dtype=float)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError('scale must be positive finite Z,Y,X spacing in micrometres')
    depth, height, width = patch_shape
    diagonal = int(np.ceil(np.hypot(height, width)))
    context = np.array([depth, diagonal, diagonal])
    halo = np.ceil((spine_radius_um + neck_radius_um) / spacing).astype(int) + 2
    margin = np.maximum(context, halo)
    slices = []
    for axis in range(3):
        projection = np.any(mask, axis=tuple(i for i in range(3) if i != axis))
        indices = np.flatnonzero(projection)
        if indices.size == 0:
            return None
        slices.append(slice(max(0, int(indices[0] - margin[axis])),
                            min(mask.shape[axis], int(indices[-1] + margin[axis] + 1))))
    return tuple(slices)


def rotation_geometry(shape, angle):
    """Same shape, matrix and centre convention as scipy.ndimage.rotate."""
    c, s = special.cosdg(angle), special.sindg(angle)
    matrix = np.array([[c, s], [-s, c]])
    iy, ix = shape[1:]
    bounds = matrix @ [[0, 0, iy, iy], [0, ix, 0, ix]]
    plane = (np.ptp(bounds, axis=1) + .5).astype(int)
    offset = (np.asarray(shape[1:]) - 1) / 2 - matrix @ ((plane - 1) / 2)
    return (shape[0], int(plane[0]), int(plane[1])), matrix, offset


def rotated_fragment(volume, start, patch_shape, geometry, order):
    """Sample only one output tile from a virtual rotated volume.

    The original source plane supplies all interpolation context. There is no
    locally rotated, clipped input tile and no allocated full rotated volume.
    """
    shape, matrix, offset = geometry
    lo = np.maximum(start, 0).astype(int)
    hi = np.minimum(np.asarray(start) + patch_shape, shape).astype(int)
    patch = np.zeros(patch_shape, dtype=volume.dtype)
    if np.any(hi <= lo):
        return patch, None
    out_lo = lo - start
    out_hi = out_lo + hi - lo
    borders = np.stack((out_lo, out_hi), axis=1).astype(np.int32)
    destination = tuple(slice(int(a), int(b)) for a, b in borders)
    # Per-plane transform preserves scipy.rotate's order=0/1 semantics in Z.
    local_offset = offset + matrix @ lo[1:]
    plane_shape = tuple(int(v) for v in (hi - lo)[1:])
    for source_z, output_z in zip(range(lo[0], hi[0]), range(out_lo[0], out_hi[0])):
        affine_transform(volume[source_z], matrix, local_offset,
                         output_shape=plane_shape,
                         output=patch[output_z, destination[1], destination[2]],
                         order=order, mode='constant', cval=0, prefilter=False)
    return patch, borders
