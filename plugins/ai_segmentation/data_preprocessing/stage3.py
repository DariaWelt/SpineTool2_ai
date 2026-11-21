import os
import json
import numpy as np
import argparse
import random
import cv2
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool, cpu_count

from tifffile import imread
from scipy.ndimage import gaussian_filter
from data_preprocessing.stage3_patches import foreground_crop, rotation_geometry, rotated_fragment
from data_preprocessing.stage3_targets import build_local_targets

from data_preprocessing.stage1 import normalize_minmax
from data_preprocessing.stage2 import calculate_interim_image, find_optimal_j0, calculate_isi
# from stage1 import normalize_minmax
# from stage2 import calculate_interim_image, find_optimal_j0, calculate_isi
FRAGMENT_SHAPE = [64, 128, 128] 
FINAL_SHAPE = [64, 128, 128]
BASE_SCALE = [0.1, 0.1]

ANGLES_POOL = [i * 10 for i in range(36)]
ANGLES_COUNT_TRAIN = 3
POSITIONS = [[0, 0, 0], [-32, -64, -64]]

def scale_to_shape(image, target_shape):

    target_z, target_y, target_x = target_shape
    
    # Если размеры уже совпадают, возвращаем
    if image.shape[1] == target_y and image.shape[2] == target_x:
        return image
    interp = cv2.INTER_LINEAR 
    
    resized = np.zeros((image.shape[0], target_y, target_x), dtype=np.float32)
    for i in range(image.shape[0]):
        resized[i] = cv2.resize(image[i].astype(np.float32), (target_x, target_y), interpolation=interp)
    return resized

def apply_gaussian_3d_stack(image, sigma=2.0):
    """Применяет Гаусс сглаживание к каждому срезу стека (для предотвращения артефактов)"""
    blurred = np.zeros_like(image, dtype=np.float32)
    for i in range(image.shape[0]):
        blurred[i] = gaussian_filter(image[i], sigma=sigma)
    return blurred

def pad_to_min_shape(volume):
    current = volume.shape
    min_z, min_y, min_x = FINAL_SHAPE
    pad_z, pad_y, pad_x = max(0, min_z - current[0]), max(0, min_y - current[1]), max(0, min_x - current[2])
    if pad_z == 0 and pad_y == 0 and pad_x == 0: return volume
    pads = [(0, pad_z), (0, pad_y), (0, pad_x)]
    return np.pad(volume, pads, mode='constant', constant_values=0)

def get_fragment(image, start_pos):
    """
    Вырезает патч 64x128x128. 
    Если выходит за границы или start_pos отрицательный -> заливает нулями.
    """
    shape = FINAL_SHAPE
    d, h, w = image.shape

    z_min, y_min, x_min = start_pos
    z_max, y_max, x_max = z_min + shape[0], y_min + shape[1], x_min + shape[2]

    img_z_min, img_y_min, img_x_min = max(0, z_min), max(0, y_min), max(0, x_min)
    img_z_max, img_y_max, img_x_max = min(d, z_max), min(h, y_max), min(w, x_max)

    patch = np.zeros(shape, dtype=image.dtype)

    out_z_min = img_z_min - z_min
    out_y_min = img_y_min - y_min
    out_x_min = img_x_min - x_min

    out_z_max = out_z_min + (img_z_max - img_z_min)
    out_y_max = out_y_min + (img_y_max - img_y_min)
    out_x_max = out_x_min + (img_x_max - img_x_min)

    # Если пересечения нет
    if out_z_max <= out_z_min or out_y_max <= out_y_min or out_x_max <= out_x_min:
        return patch, None
    
    patch[out_z_min:out_z_max, out_y_min:out_y_max, out_x_min:out_x_max] = \
        image[img_z_min:img_z_max, img_y_min:img_y_max, img_x_min:img_x_max]
    
    borders = np.array([
        [out_z_min, out_z_max], 
        [out_y_min, out_y_max], 
        [out_x_min, out_x_max]
    ], dtype=np.int32)

    return patch, borders

def split_and_save(img_c, l1_s, l2_s, isi_s, mask, name, angle, start_pos,
                   output_dir, local_maps, crop_origin=(0, 0, 0)):
    geometry = rotation_geometry(mask.shape, angle)
    counts = np.maximum(1, np.ceil((np.asarray(geometry[0]) - start_pos) / FRAGMENT_SHAPE).astype(int))
    saved_count = 0
    for k in range(counts[0]):
        for j in range(counts[1]):
            for i in range(counts[2]):
                position = np.asarray(start_pos) + np.array([k, j, i]) * FRAGMENT_SHAPE
                if angle == 0:
                    extract = lambda volume, order: get_fragment(volume, position)
                else:
                    extract = lambda volume, order: rotated_fragment(volume, position, FINAL_SHAPE, geometry, order)
                target, borders = extract(mask, 0)
                if borders is None or np.sum(target) < 100:
                    continue
                # Only accepted tiles allocate intensity channels and auxiliary maps.
                image = np.empty((4, *FINAL_SHAPE), dtype=np.float32)
                for channel, volume in enumerate((img_c, isi_s, l1_s, l2_s)):
                    image[channel] = extract(volume, 1)[0]
                aux = {key: extract(value, 0)[0] for key, value in local_maps.items()}
                fname = f"{name}_a{angle}_p{start_pos[0]}{start_pos[1]}{start_pos[2]}_idx{k}-{j}-{i}.npz"
                np.savez_compressed(os.path.join(output_dir, fname), image=image, mask=target,
                                    borders=borders, stage3_format_version=np.int32(2),
                                    crop_origin=np.asarray(crop_origin, dtype=np.int32), **aux)
                saved_count += 1
                del image, aux, target
    return saved_count


def process_item(item, output_dir, is_train_mode, base_path, preds_stage1_dir,
                 preds_stage2_dir, spine_radius_um=0.3, neck_radius_um=0.15):
    np.random.seed(); random.seed()
    name = item['name']
    volume_output_dir = os.path.join(output_dir, name)
    os.makedirs(volume_output_dir, exist_ok=True)

    def resolve(path):
        return os.path.normpath(os.path.join(base_path, path)) if path else None

    def load_binary(path):
        return imread(path) > 0

    img_path = resolve(item.get('image'))
    if not img_path or not os.path.exists(img_path):
        return 0
    s_path = resolve(item.get('spine_mask') or item.get('spines_mask'))
    h_path = resolve(item.get('shaft_mask'))
    if not s_path or not h_path or not os.path.exists(s_path) or not os.path.exists(h_path):
        raise ValueError(f"{name}: manual spine_mask (or spines_mask) and shaft_mask required")
    spine, shaft = load_binary(s_path), load_binary(h_path)
    g_path = resolve(item.get('general_mask'))
    mask = load_binary(g_path) if g_path and os.path.exists(g_path) else spine | shaft
    if spine.shape != mask.shape or shaft.shape != mask.shape:
        raise ValueError(f"{name}: manual masks must match the original image shape")
    area_path = resolve(item.get('area_of_interest'))
    area = load_binary(area_path) if area_path and os.path.exists(area_path) else None
    if area is not None:
        if area.shape != mask.shape:
            raise ValueError(f"{name}: AOI shape mismatch")
        mask &= area
    scale = item.get('scale')
    crop = foreground_crop(mask, scale, spine_radius_um, neck_radius_um, FINAL_SHAPE)
    if crop is None:
        return 0
    full_shape = mask.shape
    origin = tuple(part.start for part in crop)
    # Copies release the full TIFF buffers: views would keep them alive.
    mask = mask[crop].copy()
    spine = spine[crop].copy()
    shaft = shaft[crop].copy()
    valid = np.ones(mask.shape, dtype=np.uint8) if area is None else area[crop].astype(np.uint8)
    spine &= valid > 0
    shaft &= valid > 0

    l1_path = os.path.join(preds_stage1_dir, f"{name}_L_dendrite.tif")
    l2_path = os.path.join(preds_stage2_dir, f"{name}_necks.tif")
    if not os.path.exists(l1_path) or not os.path.exists(l2_path):
        print(f"Skipping {name}: L1 or L2 predictions missing.")
        return 0
    raw_img = imread(img_path)
    if raw_img.shape != full_shape:
        raise ValueError(f"{name}: manual masks must match the original image shape")
    if area is not None:
        raw_img *= area
    del area
    l1_raw = imread(l1_path)
    # Global intensity statistics, resizing, j0 and normalization stay unchanged.
    # Cropping before these would shift the train/inference input distribution.
    i_interim = calculate_interim_image(raw_img, scale)
    j0 = find_optimal_j0(i_interim, (l1_raw > 0.5).astype(np.uint8))
    I_si = calculate_isi(i_interim, j0)
    del i_interim
    minimum, maximum = np.min(raw_img), np.max(raw_img)
    I_s = (((raw_img[crop] - minimum) / (maximum - minimum)).astype(np.float32)
           if maximum > minimum else np.zeros(mask.shape, dtype=np.float32))
    del raw_img

    def prepare_aux(volume, smooth):
        up = scale_to_shape(volume, full_shape)
        del volume
        # Keep XY Gaussian context so the retained crop equals full-image smoothing.
        if smooth:
            halo = 8  # scipy Gaussian truncate=4, sigma=2
            expanded = (crop[0], slice(max(0, crop[1].start - halo), min(full_shape[1], crop[1].stop + halo)),
                        slice(max(0, crop[2].start - halo), min(full_shape[2], crop[2].stop + halo)))
            local = up[expanded].astype(np.float32, copy=True)
            del up
            filtered = apply_gaussian_3d_stack(local, sigma=2.0)
            del local
            interior = (slice(None), slice(crop[1].start - expanded[1].start, crop[1].stop - expanded[1].start),
                        slice(crop[2].start - expanded[2].start, crop[2].stop - expanded[2].start))
            return filtered[interior].copy()
        return up[crop].astype(np.float32, copy=True)

    l1_s = prepare_aux(l1_raw, True)
    del l1_raw
    isi_s = prepare_aux(I_si, True)
    del I_si
    l2_s = prepare_aux(imread(l2_path), False)
    print(f"{name}: crop {full_shape} -> {mask.shape}, origin={origin}", flush=True)
    # EDT and skeletonization now see only the foreground crop and physical halo.
    local_maps = build_local_targets(mask, spine, shaft, scale, spine_radius_um, neck_radius_um)
    del spine, shaft
    local_maps['valid_mask'] = valid
    del valid
    I_s, l1_s, l2_s, isi_s = (pad_to_min_shape(v) for v in (I_s, l1_s, l2_s, isi_s))
    mask = pad_to_min_shape(mask.astype(np.uint8))
    local_maps = {key: pad_to_min_shape(value) for key, value in local_maps.items()}
    angles = random.sample(ANGLES_POOL, ANGLES_COUNT_TRAIN) if is_train_mode else [0]
    total_saved = 0
    for angle in angles:
        for pos in POSITIONS:
            if not is_train_mode and pos != [0, 0, 0]:
                continue
            total_saved += split_and_save(I_s, l1_s, l2_s, isi_s, mask, name, angle, pos,
                                          volume_output_dir, local_maps, origin)
    return total_saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--l1_dir", required=True, help="Путь к предсказаниям L1 (Дендриты)")
    parser.add_argument("--l2_dir", required=True, help="Путь к предсказаниям L2 (Шейки/Дендриты)")
    parser.add_argument("--mode", required=True, choices=["train", "val", "test"])
    parser.add_argument("--base_path", default="../../")
    parser.add_argument("--workers", type=int, default=min(2, cpu_count()))
    parser.add_argument('--spine_radius_um', type=float, default=0.3)
    parser.add_argument('--neck_radius_um', type=float, default=0.15)
    parser.add_argument('--local_masks_json', help='Stage4 JSON with manual spine_mask and shaft_mask, matched by name')
    parser.add_argument('--local_masks_base_path', help='Base directory for paths in local_masks_json; defaults to base_path')
    parser.add_argument('--maxtasksperchild', type=int, default=1,
                        help='Recycle worker after this many images to return memory to OS')
    args = parser.parse_args()
    if args.workers < 1 or args.maxtasksperchild < 1:
        parser.error('workers and maxtasksperchild must be >= 1')
    
    final_output_dir = os.path.join(args.out_dir, args.mode)
    os.makedirs(final_output_dir, exist_ok=True)
    random.seed(42)
    
    with open(args.json_path) as f: cfg = json.load(f)
    if args.local_masks_json:
        with open(args.local_masks_json) as f:
            local_cfg = json.load(f)
        local_items = {entry['name']: entry for entry in local_cfg['data']}
        local_base = args.local_masks_base_path or args.base_path
        for entry in cfg['data']:
            local_entry = local_items.get(entry['name'], {})
            for key in ('spine_mask', 'spines_mask', 'shaft_mask'):
                if local_entry.get(key):
                    entry[key] = os.path.abspath(os.path.join(local_base, local_entry[key]))
    items = []
    ts, vs = set(cfg.get('test_data', [])), set(cfg.get('validation_data', []))
    for x in cfg['data']:
        n = x['name']
        if args.mode == 'train' and n not in ts and n not in vs: items.append(x)
        elif args.mode == 'val' and n in vs: items.append(x)
        elif args.mode == 'test' and n in ts: items.append(x)
        
    print(f"Mode: {args.mode}. Items: {len(items)}")
    worker_func = partial(
        process_item, 
        output_dir=final_output_dir, 
        is_train_mode=(args.mode == 'train'), 
        base_path=args.base_path,
        preds_stage1_dir=args.l1_dir,
        preds_stage2_dir=args.l2_dir,
        spine_radius_um=args.spine_radius_um, neck_radius_um=args.neck_radius_um
    )
    
    total_patches = 0
    with Pool(processes=args.workers, maxtasksperchild=args.maxtasksperchild) as pool:
        for count in tqdm(pool.imap_unordered(worker_func, items), total=len(items)):
            total_patches += count
    print(f" Generated {total_patches} patches (128x128x64) for Stage 3.")
