import json
import re
import numpy as np
from PIL import Image
import cv2
from urllib.parse import urlparse
import os
from aiohttp import ClientTimeout
import asyncio
import aiohttp
import copy
import sys
import time
import threading
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import psutil
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from warp_image.tps import numpy as tps
from scipy.ndimage import distance_transform_edt, gaussian_filter


ROOT_DIR = "/home/dev/code/test-color-mockup-2d"
PREFIX = "optimized-mockup-infos/mockups/all_over_print_full_zip_up_hoodie_lightweight"


def generate_local_path_from_url(url, prefix):
    url_parsed = urlparse(url)
    url_path = url_parsed.path.lstrip('/')
    return os.path.join(ROOT_DIR, prefix, url_path)


async def download_url(url, prefix, session, max_retries=5):
    local_path = generate_local_path_from_url(url, prefix)
    
    if os.path.exists(local_path):
        print(f"Cached: {local_path}")
        return local_path
    
    for attempt in range(max_retries):
        try:
            async with session.get(url, timeout=ClientTimeout(total=30)) as response:
                response.raise_for_status()
                
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                
                with open(local_path, 'wb') as f:
                    async for chunk in response.content.iter_chunked(8192):
                        if chunk:
                            f.write(chunk)
                print(f"Downloaded: {local_path}")
                return local_path
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Retrying {url} ({attempt + 1}/{max_retries}): {str(e)}")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
            else:
                print(f"Error downloading {url} after {max_retries} attempts: {str(e)}")
                return False
    return False


async def bulk_download(urls, prefix, max_concurrent=10):
    """Download multiple files concurrently with a limit."""
    async with aiohttp.ClientSession() as session:
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def bounded_download(url, prefix):
            async with semaphore:
                return await download_url(url, prefix, session)
        
        tasks = [bounded_download(url, prefix) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results


def extract_urls_with_paths(obj, path=[]):
    """
    Recursively extract all URLs from the mockup_infos object.
    Returns: list of tuples (url, path_to_url_in_object)
    """
    urls_with_paths = []
    
    if isinstance(obj, dict):
        for key, value in obj.items():
            urls_with_paths.extend(extract_urls_with_paths(value, path + [key]))
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            urls_with_paths.extend(extract_urls_with_paths(item, path + [idx]))
    elif isinstance(obj, str):
        # Simple URL detection (http/https)
        if obj.startswith(('http://', 'https://')):
            urls_with_paths.append((obj, path))
    
    return urls_with_paths


def set_nested_value(obj, path, value):
    """Set a value in a nested object using a path list."""
    current = obj
    for key in path[:-1]:
        if isinstance(current, dict):
            current = current[key]
        elif isinstance(current, list):
            current = current[int(key)]
    
    final_key = path[-1]
    if isinstance(current, dict):
        current[final_key] = value
    elif isinstance(current, list):
        current[int(final_key)] = value


async def download_mockup_infos(mockup_infos):
    """
    Download all URLs from mockup_infos and replace them with local paths.
    
    Args:
        mockup_infos: dict or list containing URLs to download
        prefix: directory prefix for saving downloaded files
    
    Returns:
        str: path to the updated mockup_infos JSON file
    """
    # Extract all URLs and their positions in the object
    urls_with_paths = extract_urls_with_paths(mockup_infos)
    
    if not urls_with_paths:
        print("No URLs found in mockup_infos")
        return None
    
    print(f"Found {len(urls_with_paths)} URLs to download")
    
    # Extract unique URLs
    unique_urls = list(set(url for url, _ in urls_with_paths))
    
    # Download all files concurrently
    download_results = await bulk_download(unique_urls, f"{PREFIX}/downloads")
    
    # Map original URLs to their local paths
    url_to_local_path = {}
    for url, result in zip(unique_urls, download_results):
        if isinstance(result, str):  # Successful download
            url_to_local_path[url] = result
        else:
            print(f"Failed to download {url}")
    
    # Create a deep copy of mockup_infos to avoid modifying the original
    updated_infos = copy.deepcopy(mockup_infos)
    
    # Replace all URLs with their local paths
    for url, path in urls_with_paths:
        if url in url_to_local_path:
            local_path = url_to_local_path[url]
            set_nested_value(updated_infos, path, local_path)
    
    # Save updated mockup_infos to a JSON file
    output_dir = os.path.join(ROOT_DIR, PREFIX)
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "mockup_infos.downloaded.json")
    with open(output_path, 'w') as f:
        json.dump(updated_infos, f, indent=2)
    
    print(f"Updated mockup_infos saved to: {output_path}")
    return output_path


def process_model_parts(side_name, parts):
    base = Image.new('RGBA', (1000,1000), (0, 0, 0, 0))
    for part in parts:
        img = Image.open(part["image_path"]).convert('RGBA')
        base.paste(img, (0, 0), img)
    
    path = f"{ROOT_DIR}/{PREFIX}/optimized/{side_name.lower()}.model.png"
    base.save(path)
    
    return {
        "name": f"{side_name}.Model",
        "image_path": path
    }


def remove_noise_pixels(mask_path, min_area=100, close_kernel=3):
    mask_img = Image.open(mask_path).convert('RGBA')
    mask_array = np.array(mask_img)
    alpha_channel = mask_array[:, :, 3]
    
    binary_mask = (alpha_channel > 0).astype(np.uint8) * 255
    
    # Morphological closing để lấp gaps nhỏ trước khi tìm contour
    if close_kernel > 0:
        kernel = np.ones((close_kernel, close_kernel), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    clean_mask = np.zeros_like(binary_mask)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            cv2.drawContours(
                clean_mask, [contour], -1, 255, 
                thickness=cv2.FILLED
            )
    
    return clean_mask > 0


def join_model_data(base, model, mask_path, min_noise_area=100):
    mask_img = Image.open(mask_path).convert('RGBA')
    mask_array = np.array(mask_img).astype(np.float32)
    mask_alpha = mask_array[:, :, 3] / 255.0
    
    # Chỉ dùng noise removal để zero out các blob nhỏ li ti
    # KHÔNG convert toàn bộ về binary
    noise_mask = remove_noise_pixels(mask_path, min_area=min_noise_area)
    mask_alpha[~noise_mask] = 0.0  # zero out noise pixels, giữ nguyên phần còn lại
    
    for ch in range(base.shape[2]):
        base_ch = base[:, :, ch]
        model_ch = model[:, :, ch]
        base_nan = np.isnan(base_ch)

        # Winner-takes-all: không blend tọa độ, chỉ pick coord từ 1 model.
        # Masks gần như binary (>99.5%), warp liên tục tại seam (<1px diff)
        # nên pick coord từ model nào có alpha cao hơn là đủ chính xác.
        blended = np.where(
            base_nan,
            np.where(mask_alpha > 0, model_ch, np.nan),
            np.where(mask_alpha > 0.5, model_ch, base_ch)
        )
        base[:, :, ch] = blended


def resize_model(model_data, mask, target_size):
    """
    Resize với soft edge để tránh jagged artifacts
    """
    
    # 1. Fill NaN
    filled_data = model_data.copy()
    
    if len(model_data.shape) == 3:
        for i in range(model_data.shape[2]):
            channel = model_data[:, :, i].copy()
            nan_mask = np.isnan(channel)
            
            if nan_mask.any() and mask.any():
                indices = distance_transform_edt(
                    nan_mask, 
                    return_distances=False, 
                    return_indices=True
                )
                filled_data[:, :, i] = channel[tuple(indices)]
    else:
        nan_mask = np.isnan(model_data)
        if nan_mask.any() and mask.any():
            indices = distance_transform_edt(
                nan_mask,
                return_distances=False,
                return_indices=True
            )
            filled_data = model_data[tuple(indices)]
    
    # 2. Tạo soft mask (blur mask trước khi resize)
    soft_mask = gaussian_filter(mask.astype(np.float32), sigma=2.0)
    
    # 3. Apply soft mask lên data TRƯỚC khi resize
    if len(filled_data.shape) == 3:
        for i in range(filled_data.shape[2]):
            filled_data[:, :, i] = filled_data[:, :, i] * soft_mask
    else:
        filled_data = filled_data * soft_mask
    
    # 4. Resize
    if len(filled_data.shape) == 3:
        resized = cv2.resize(
            filled_data.astype(np.float32),
            (target_size[1], target_size[0]),
            interpolation=cv2.INTER_AREA
        )
    else:
        resized = cv2.resize(
            filled_data.astype(np.float32),
            (target_size[1], target_size[0]),
            interpolation=cv2.INTER_AREA
        )
    
    # 5. Resize mask (hard threshold)
    resized_mask = cv2.resize(
        mask.astype(np.uint8) * 255,
        (target_size[1], target_size[0]),
        interpolation=cv2.INTER_AREA
    ) > 0
    
    # 6. Set NaN - nhưng GIỮ soft edge
    # Tạo transition zone
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(resized_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    edge_zone = resized_mask & ~eroded
    
    if len(resized.shape) == 3:
        for i in range(resized.shape[2]):
            # Chỉ set NaN hoàn toàn bên ngoài, giữ edge zone
            resized[:, :, i][~resized_mask] = np.nan
    else:
        resized[~resized_mask] = np.nan
            
    return resized


def _make_100x100_json(model_1000: np.ndarray, target: int = 100) -> np.ndarray:
    """
    Downsample merged 1000×1000 warp grid → 100×100 cho JS WebGL renderer.
    NaN pixels (ngoài garment silhouette) được fill bằng nearest-valid neighbor
    trước khi subsample để tránh artifact khi JS clamp NaN về 0.
    """
    H, W = model_1000.shape[:2]
    filled = model_1000.copy()

    for ch in range(2):
        channel = filled[:, :, ch]
        nan_mask = np.isnan(channel)
        if nan_mask.any() and (~nan_mask).any():
            indices = distance_transform_edt(nan_mask, return_distances=False, return_indices=True)
            filled[:, :, ch] = channel[tuple(indices)]

    # Lấy đều target×target điểm từ grid đã fill
    row_idx = np.linspace(0, H - 1, target, dtype=int)
    col_idx = np.linspace(0, W - 1, target, dtype=int)
    return filled[np.ix_(row_idx, col_idx)]


def _get_figure_key(part_name: str) -> str:
    """
    Trích figure key từ tên part.
    Tên 5 segment: Front_Back.Front_Back.Back.Design.Back_4  → "Back"
    Tên 4 segment: Front.Front.Design.Front_2                → "default"
    """
    segs = part_name.split(".")
    return segs[2] if len(segs) == 5 else "default"


def _merge_parts_for_figure(file_key: str, parts: list) -> dict:
    """
    Collect per-part warp data for a figure.

    No coordinate merging — each part keeps its own TPS grid.
    Rendering uses per-part warp + Porter-Duff composite to avoid
    coordinate-discontinuity / jagged-edge artifacts at seam regions.
    """
    out_dir   = f"{ROOT_DIR}/{PREFIX}/optimized"
    orig_size = (1000, 1000)

    merged_alpha = np.zeros(orig_size, dtype=np.float32)
    parts_data   = []

    for part in parts:
        npy_path  = part["warp_info"]["model"]
        mask_path = part["mask_path"]

        part_alpha   = np.array(Image.open(mask_path).convert('RGBA'))[:, :, 3] / 255.0
        merged_alpha = part_alpha + merged_alpha * (1.0 - part_alpha)

        part_npy = np.load(npy_path, allow_pickle=True)
        grid_100 = _make_100x100_json(part_npy)
        safe_name     = re.sub(r'[^A-Za-z0-9._-]', '_', part["name"])
        json_path     = f"{out_dir}/{safe_name}.100x100.json"
        with open(json_path, 'w') as f:
            json.dump(grid_100.tolist(), f, separators=(",", ":"))

        parts_data.append({
            "model":      npy_path,   # full 1000×1000 npy (Python renderer)
            "model_json": json_path,  # 100×100 json       (JS renderer)
            "mask_path":  mask_path,
        })

    merged_alpha = np.clip(merged_alpha, 0.0, 1.0)
    mask_arr = np.zeros((*orig_size, 4), dtype=np.uint8)
    mask_arr[:, :, 3] = (merged_alpha * 255).astype(np.uint8)
    merged_mask_path = f"{out_dir}/{file_key}.warp_mask.png"
    Image.fromarray(mask_arr, 'RGBA').save(merged_mask_path)

    return {
        "name":      f"{file_key}.Design",
        "side":      "front",
        "mask_path": merged_mask_path,
        "warp_type": "warp_npy",
        "warp_info": {
            "parts":          parts_data,
            "artwork_width":  3000,
            "artwork_height": 3000,
        },
        "effects": [],
        "fill":    100,
        "opacity": 100,
    }


def process_design_parts(side_name, parts) -> list:
    """
    Group design parts by figure, merge mỗi figure độc lập.
    Trả list vì 1 view có thể có nhiều figure (ví dụ Front_Back có Back + Front).
    Parts của 2 figure khác nhau không được gộp chung vì chúng map tới
    các vùng hoàn toàn khác nhau của artwork (cách nhau 1000+ px).
    """
    # Group theo figure key, giữ thứ tự xuất hiện
    groups: dict[str, list] = {}
    for part in parts:
        key = _get_figure_key(part["name"])
        groups.setdefault(key, []).append(part)

    result = []
    for figure_key, fig_parts in groups.items():
        if figure_key == "default":
            file_key = side_name.lower()
        else:
            file_key = f"{side_name.lower()}.{figure_key.lower()}"
        result.append(_merge_parts_for_figure(file_key, fig_parts))

    return result


def _upsample_grid_seam_aware(grid_small: np.ndarray, canvas_size: tuple) -> np.ndarray:
    """
    Upsample sparse grid (e.g. 100×100) lên canvas_size dùng hybrid interpolation:
    - INTER_LINEAR cho vùng smooth bên trong panel (cho kết quả mượt)
    - INTER_NEAREST cho vùng seam giữa các panel (tránh blend tọa độ từ 2 panel khác nhau)

    Seam được detect trực tiếp từ grid: hai grid cell cạnh nhau có coordinate jump
    lớn hơn SEAM_THRESH là seam — không cần lưu thêm metadata vào JSON.

    Tại sao không dùng INTER_LINEAR đơn thuần: tại seam giữa 2 panel, tọa độ nhảy
    đột ngột (có thể 680px trong artwork), bilinear blend 2 điểm đó tạo ra tọa độ
    hoàn toàn sai → màu sắc sai trong vùng ~10px quanh mỗi seam.
    """
    W, H = canvas_size  # cv2 convention: (width, height)

    # Detect seam edges: adjacent cells differ > threshold in normalized coords
    # 0.02 normalized = 20px in 1000px artwork, đủ để bắt panel seams (~680px)
    # mà không trigger trên smooth warp variations (thường < 0.005)
    SEAM_THRESH = 0.02
    dx = np.abs(np.diff(grid_small, axis=1)).max(axis=2)  # (GH, GW-1)
    dy = np.abs(np.diff(grid_small, axis=0)).max(axis=2)  # (GH-1, GW)

    # Mark cả 2 cells hai bên mỗi seam edge là seam-adjacent
    GH, GW = grid_small.shape[:2]
    cell_seam = np.zeros((GH, GW), dtype=np.uint8)
    cell_seam[:, :-1] |= (dx > SEAM_THRESH)
    cell_seam[:, 1:]  |= (dx > SEAM_THRESH)
    cell_seam[:-1, :] |= (dy > SEAM_THRESH)
    cell_seam[1:, :]  |= (dy > SEAM_THRESH)

    grid_linear  = cv2.resize(grid_small, (W, H), interpolation=cv2.INTER_LINEAR)
    grid_nearest = cv2.resize(grid_small, (W, H), interpolation=cv2.INTER_NEAREST)
    # Upsample seam mask với NEAREST để giữ sharp boundary
    seam_pixels  = cv2.resize(cell_seam,  (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)

    return np.where(seam_pixels[:, :, None], grid_nearest, grid_linear)


def warp_image_per_parts(image, parts: list, artwork_size: tuple, warped_size: tuple,
                         use_json: bool = False) -> Image.Image:
    """
    Per-part warp + Porter-Duff alpha composite.

    Each sub-part is warped independently using its own TPS grid (smooth, no
    discontinuities) and composited onto the canvas in order.  This is the
    only approach that avoids seam / jagged-edge artifacts at part boundaries.

    Root cause of artifacts with coordinate-merging:
      Adjacent canvas pixels assigned to different parts have artwork
      coordinates that can differ by 1 000+ px → huge color jump → staircase.

    use_json=False  → load full 1000×1000 npy (high quality, slower)
    use_json=True   → load 100×100 JSON then bilinear-upsample (fast, JS-parity)
    """
    if isinstance(image, Image.Image):
        image = np.array(image)

    W, H = warped_size
    resized_img = cv2.resize(image, artwork_size, cv2.INTER_LANCZOS4)

    canvas = np.zeros((H, W, 4), dtype=np.float32)

    for part in parts:
        # ── load mask (shared by both branches below) ────────────────────
        mask_a = np.array(Image.open(part["mask_path"]).convert("RGBA"))[:, :, 3].astype(
            np.float32) / 255.0

        # ── load grid ────────────────────────────────────────────────────
        if use_json:
            with open(part["model_json"]) as f:
                grid = np.array(json.load(f), dtype=np.float32)
            # Bilinear upsample: smooth within the part, but the pre-filled
            # NaN zone in the 100×100 JSON bleeds wrong coordinates into
            # boundary pixels after interpolation.
            # Fix: re-NaN outside the mask then fill from within-mask only.
            grid = cv2.resize(grid, (W, H), interpolation=cv2.INTER_LINEAR)
            outside = mask_a < 0.1
            for ch in range(2):
                grid[:, :, ch][outside] = np.nan
            for ch in range(2):
                nan_m = np.isnan(grid[:, :, ch])
                if nan_m.any() and (~nan_m).any():
                    idx = distance_transform_edt(nan_m, return_distances=False,
                                                 return_indices=True)
                    grid[:, :, ch] = grid[:, :, ch][tuple(idx)]
        else:
            grid = np.load(part["model"], allow_pickle=True)
            # Fill NaN outside part boundary so cv2.remap never reads garbage
            for ch in range(2):
                nan_m = np.isnan(grid[:, :, ch])
                if nan_m.any() and (~nan_m).any():
                    idx = distance_transform_edt(nan_m, return_distances=False,
                                                 return_indices=True)
                    grid[:, :, ch] = grid[:, :, ch][tuple(idx)]

        # ── warp ─────────────────────────────────────────────────────────
        mapx, mapy = tps.tps_grid_to_remap(grid, artwork_size)
        warped = cv2.remap(resized_img, mapx, mapy, cv2.INTER_CUBIC).astype(np.float32)

        # ── apply part mask ───────────────────────────────────────────────
        warped[:, :, 3] = np.minimum(warped[:, :, 3], mask_a * 255.0)

        # ── Porter-Duff "src over dst" ────────────────────────────────────
        src_a = warped[:, :, 3:4] / 255.0
        dst_a = canvas[:, :, 3:4] / 255.0
        out_a = src_a + dst_a * (1.0 - src_a)
        safe_a = np.where(out_a > 1e-6, out_a, 1.0)
        canvas[:, :, :3] = (
            warped[:, :, :3] * src_a +
            canvas[:, :, :3] * dst_a * (1.0 - src_a)
        ) / safe_a
        canvas[:, :, 3] = out_a[:, :, 0] * 255.0

    return Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))


def warp_image(image, model_path, artwork_size, warped_size):
    """
    Warp image using TPS (Thin Plate Spline) transformation.
    Hỗ trợ cả .npy (full grid) và .json (sparse grid, được upsample bằng
    bilinear interpolation để match cách JS WebGL renderer xử lý).
    """
    if isinstance(image, Image.Image):
        image = np.array(image)

    resized_img = cv2.resize(image, artwork_size, cv2.INTER_LANCZOS4)

    if model_path.endswith('.json'):
        with open(model_path) as f:
            grid = np.array(json.load(f), dtype=np.float32)
        grid = _upsample_grid_seam_aware(grid, warped_size)
    else:
        grid = np.load(model_path, allow_pickle=True)

    mapx, mapy = tps.tps_grid_to_remap(grid, artwork_size)
    
    # Apply the warping transformation
    img = cv2.remap(np.array(resized_img), mapx, mapy, cv2.INTER_CUBIC)
    
    return Image.fromarray(img).resize(warped_size, Image.LANCZOS)


def apply_blend_mode(base, overlay, blend_mode):
    """
    Apply blend mode to combine two images
    
    Args:
        base: base image as numpy array (RGBA)
        overlay: overlay image as numpy array (RGBA)
        blend_mode: str - 'normal', 'multiply', 'screen', 'linear_dodge'
    
    Returns:
        Combined image as numpy array
    """
    # Extract alpha channels
    base_alpha = base[:, :, 3:4] / 255.0
    overlay_alpha = overlay[:, :, 3:4] / 255.0
    
    # Extract RGB channels
    base_rgb = base[:, :, :3].astype(np.float32)
    overlay_rgb = overlay[:, :, :3].astype(np.float32)
    
    if blend_mode == 'multiply':
        blended = (base_rgb * overlay_rgb) / 255.0
    elif blend_mode == 'screen':
        blended = 255 - ((255 - base_rgb) * (255 - overlay_rgb)) / 255.0
    elif blend_mode == 'linear_dodge':
        blended = np.clip(base_rgb + overlay_rgb, 0, 255)
    else:  # normal
        blended = overlay_rgb
    
    # Blend with alpha
    result_rgb = base_rgb * (1 - overlay_alpha) + blended * overlay_alpha
    result_alpha = np.clip((base_alpha + overlay_alpha * (1 - base_alpha)) * 255, 0, 255)
    
    result = np.dstack([result_rgb.astype(np.uint8), result_alpha.astype(np.uint8)])
    return result


def generate_mockup(mockup_info, artwork_path, output_path, mask_mode=False, use_json_model=False):
    """
    Generate a mockup image from mockup_info and artwork.
    use_json_model=True → dùng warp_info['model_json'] (100x100) thay vì 'model' (1000x1000 npy).
    """
    size = mockup_info.get('size', {'width': 1000, 'height': 1000})
    canvas_size = (size['width'], size['height'])

    canvas = Image.new('RGBA', canvas_size, (0, 0, 0, 0))
    artwork = Image.open(artwork_path).convert('RGBA')

    for part in mockup_info['parts']:
        part_name = part['name']

        if 'image_path' in part and part.get('warp_type') != 'warp_npy':
            layer = Image.open(part['image_path']).convert('RGBA')

            if 'blend_mode' in part and part['blend_mode'] != 'normal':
                canvas_array = np.array(canvas)
                layer_array = np.array(layer)
                blended = apply_blend_mode(canvas_array, layer_array, part['blend_mode'])
                canvas = Image.fromarray(blended, 'RGBA')
            else:
                canvas.paste(layer, (0, 0), layer)

        elif part.get('warp_type') == 'warp_npy':
            warp_info    = part['warp_info']
            artwork_size = (warp_info['artwork_width'], warp_info['artwork_height'])

            if 'parts' in warp_info:
                # Per-part warp + composite — seam-free (new format)
                warped = warp_image_per_parts(
                    artwork, warp_info['parts'], artwork_size, canvas_size,
                    use_json=use_json_model,
                )
            elif 'model' in warp_info:
                # Legacy: single merged npy (may have seam artifacts)
                warped = warp_image(artwork, warp_info['model'], artwork_size, canvas_size)
                if mask_mode and 'mask_path' in part:
                    mask        = Image.open(part['mask_path']).convert('RGBA')
                    warped_arr  = np.array(warped)
                    warped_arr[:, :, 3] = np.minimum(
                        warped_arr[:, :, 3],
                        np.array(mask)[:, :, 3]
                    )
                    warped = Image.fromarray(warped_arr, 'RGBA')
            else:
                continue

            # Apply opacity
            opacity = part.get('opacity', 100)
            if opacity < 100:
                warped_arr       = np.array(warped)
                warped_arr[:, :, 3] = (warped_arr[:, :, 3] * opacity / 100).astype(np.uint8)
                warped = Image.fromarray(warped_arr, 'RGBA')

            canvas.paste(warped, (0, 0), warped)
    
    # Save the result
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    canvas.save(output_path)


class ResourceMonitor:
    """Track peak CPU and memory usage during a timed block."""

    def __init__(self, interval: float = 0.1):
        self._proc = psutil.Process()
        self._interval = interval
        self._thread = None
        self._stop = threading.Event()
        self.peak_mem_mb: float = 0.0
        self.cpu_samples: list = []
        self._cpu_times_start = None
        self._wall_start: float = 0.0

    def start(self):
        self._stop.clear()
        self.peak_mem_mb = 0.0
        self.cpu_samples = []
        self._cpu_times_start = self._proc.cpu_times()
        self._wall_start = time.time()

        def _poll():
            while not self._stop.is_set():
                try:
                    mem = self._proc.memory_info().rss / 1024 / 1024
                    cpu = self._proc.cpu_percent(interval=None)
                    if mem > self.peak_mem_mb:
                        self.peak_mem_mb = mem
                    self.cpu_samples.append(cpu)
                except psutil.Error:
                    pass
                self._stop.wait(self._interval)

        # Prime cpu_percent so first sample is meaningful
        self._proc.cpu_percent(interval=None)
        self._thread = threading.Thread(target=_poll, daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        ct_end = self._proc.cpu_times()
        ct_start = self._cpu_times_start
        cpu_user  = ct_end.user  - ct_start.user
        cpu_sys   = ct_end.system - ct_start.system
        wall      = time.time() - self._wall_start
        avg_cpu   = float(np.mean(self.cpu_samples)) if self.cpu_samples else 0.0
        return {
            "peak_mem_mb":  round(self.peak_mem_mb, 1),
            "cpu_user_s":   round(cpu_user, 3),
            "cpu_sys_s":    round(cpu_sys, 3),
            "avg_cpu_pct":  round(avg_cpu, 1),
            "wall_s":       round(wall, 3),
        }


def count_warp_parts(mockup_infos):
    """Count the number of warp parts in the configuration"""
    total_warp_parts = 0
    for mockup_info in mockup_infos['mockup_infos']:
        for part in mockup_info['parts']:
            if part.get('warp_type') == 'warp_npy':
                total_warp_parts += 1
    return total_warp_parts


def generate_mockups_from_config(mockup_infos, config_type, artwork_dir, output_base_dir, log_file,
                                 use_json_model=False):
    """
    Generate mockups from a specific configuration (original or optimized).
    use_json_model=True → dùng model_json (100x100) thay vì model npy (1000x1000).
    """
    warp_parts_count = count_warp_parts(mockup_infos)

    msg = f"\n{'='*60}\nGenerating {config_type.upper()} mockups\n{'='*60}"
    print(msg)
    log_file.write(msg + "\n")

    msg = f"Configuration: {len(mockup_infos['mockup_infos'])} mockup views, {warp_parts_count} warp parts total"
    print(msg)
    log_file.write(msg + "\n")

    mockup_output_dir = os.path.join(output_base_dir, config_type)

    # Get all artwork files
    artwork_files = [f for f in os.listdir(artwork_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    if not artwork_files:
        msg = f"No artwork files found in {artwork_dir}"
        print(msg)
        log_file.write(msg + "\n")
        return 0, 0, [], warp_parts_count, {}

    monitor = ResourceMonitor()
    monitor.start()

    total_start_time = time.time()
    mockup_count = 0
    individual_times = []

    mask_mode = True

    for artwork_file in artwork_files:
        artwork_path = os.path.join(artwork_dir, artwork_file)
        artwork_name = os.path.splitext(artwork_file)[0]

        msg = f"\nProcessing artwork: {artwork_file}"
        print(msg)
        log_file.write(msg + "\n")

        for mockup_info in mockup_infos['mockup_infos']:
            mockup_name = mockup_info['name']
            output_filename = f"{artwork_name}_{mockup_name}.png"
            output_path = os.path.join(mockup_output_dir, output_filename)

            mockup_start = time.time()
            msg = f"  Generating: {mockup_name}..."
            print(msg, end=" ")
            log_file.write(msg)

            generate_mockup(mockup_info, artwork_path, output_path, mask_mode, use_json_model)

            mockup_duration = time.time() - mockup_start
            individual_times.append(mockup_duration)
            msg = f"✓ ({mockup_duration:.3f}s)"
            print(msg)
            log_file.write(" " + msg + "\n")
            mockup_count += 1

    total_time = time.time() - total_start_time
    res = monitor.stop()

    msg = (
        f"\n{'='*60}\n{config_type.upper()} Summary:\n"
        f"  Total mockups: {mockup_count}\n"
        f"  Warp parts per mockup: {warp_parts_count}\n"
        f"  Total warp operations: {mockup_count * warp_parts_count}\n"
        f"  Total time: {total_time:.3f}s\n"
        f"  Average time per mockup: {total_time/mockup_count:.3f}s\n"
        f"  Output directory: {mockup_output_dir}\n"
        f"  --- Resource Usage ---\n"
        f"  Peak memory:    {res['peak_mem_mb']:.1f} MB\n"
        f"  CPU time:       {res['cpu_user_s']:.3f}s user + {res['cpu_sys_s']:.3f}s sys\n"
        f"  Avg CPU usage:  {res['avg_cpu_pct']:.1f}%\n"
        f"  CPU efficiency: {(res['cpu_user_s'] + res['cpu_sys_s']) / res['wall_s'] * 100:.1f}% of wall time\n"
        f"{'='*60}"
    )
    print(msg)
    log_file.write(msg + "\n")

    return total_time, mockup_count, individual_times, warp_parts_count, res


def generate_comparison_chart(original_time, optimized_time, original_times, optimized_times,
                             original_warp_count, optimized_warp_count, output_path):
    """
    Generate a comparison chart showing performance metrics
    
    Args:
        original_time: total time for original
        optimized_time: total time for optimized
        original_times: list of individual mockup times for original
        optimized_times: list of individual mockup times for optimized
        original_warp_count: number of warp parts in original
        optimized_warp_count: number of warp parts in optimized
        output_path: path to save the chart
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Mockup Generation Performance Comparison', fontsize=16, fontweight='bold')
    
    # 1. Total Time Comparison (Bar Chart)
    ax1 = axes[0, 0]
    categories = ['Original', 'Optimized']
    times = [original_time, optimized_time]
    colors = ['#FF6B6B', '#4ECDC4']
    bars = ax1.bar(categories, times, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_ylabel('Time (seconds)', fontweight='bold')
    ax1.set_title('Total Generation Time', fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}s',
                ha='center', va='bottom', fontweight='bold')
    
    # 2. Average Time per Mockup (Bar Chart)
    ax2 = axes[0, 1]
    avg_original = np.mean(original_times)
    avg_optimized = np.mean(optimized_times)
    avgs = [avg_original, avg_optimized]
    bars = ax2.bar(categories, avgs, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_ylabel('Time (seconds)', fontweight='bold')
    ax2.set_title('Average Time per Mockup', fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}s',
                ha='center', va='bottom', fontweight='bold')
    
    # 3. Individual Mockup Times (Line Chart)
    ax3 = axes[1, 0]
    x = range(1, len(original_times) + 1)
    ax3.plot(x, original_times, marker='o', label='Original', color=colors[0], linewidth=2)
    ax3.plot(x, optimized_times, marker='s', label='Optimized', color=colors[1], linewidth=2)
    ax3.set_xlabel('Mockup Number', fontweight='bold')
    ax3.set_ylabel('Time (seconds)', fontweight='bold')
    ax3.set_title('Individual Mockup Generation Times', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Performance Metrics (Text Summary)
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    speedup = original_time / optimized_time if optimized_time > 0 else 0
    time_saved = original_time - optimized_time
    percent_saved = (time_saved / original_time * 100) if original_time > 0 else 0
    
    summary_text = f"""
    PERFORMANCE METRICS
    {'='*30}
    
    Original:
      • Total Time: {original_time:.3f}s
      • Average/Mockup: {avg_original:.3f}s
      • Mockups: {len(original_times)}
      • Warp Parts: {original_warp_count}
    
    Optimized:
      • Total Time: {optimized_time:.3f}s
      • Average/Mockup: {avg_optimized:.3f}s
      • Mockups: {len(optimized_times)}
      • Warp Parts: {optimized_warp_count}
    
    Improvement:
      • Speedup: {speedup:.2f}x faster
      • Time Saved: {time_saved:.3f}s
      • Reduction: {percent_saved:.1f}%
      • Parts Reduction: {original_warp_count - optimized_warp_count}
    """
    
    ax4.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round',
             facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Comparison chart saved to: {output_path}")


def process_for_side(mockup_info):
    """
    Rebuild mockup_info preserving per-figure compositing order:
      BG → [figure_1: model → colors → design → fx] → [figure_2: ...] → ...

    In multi-figure views (e.g. Front_Back), each figure has its own model photo,
    design warp, and effect layers. Mixing them breaks compositing (e.g. Back effects
    applied on top of Front design instead of before it).
    """
    side_name = mockup_info["name"]

    out_dir = f"{ROOT_DIR}/{PREFIX}/optimized"
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    PART_TYPES = {"Design", "Colors", "fx"}

    bg_parts = []
    figures = {}       # {figure_key: {'model': [], 'color': [], 'design': [], 'effect': []}}
    figure_order = []  # preserves first-seen order

    for part in mockup_info["parts"]:
        segs = part["name"].split(".")

        if segs[1] == "BG":
            bg_parts.append(part)
            continue

        # Determine figure key and part type
        if len(segs) == 5:
            # Multi-figure: SideName.SideName.FigureKey.PartType.PartName
            figure_key = segs[2]
            part_type  = segs[3]
        elif len(segs) == 4 and segs[2] not in PART_TYPES:
            # Multi-figure model: SideName.SideName.FigureKey.PartName
            figure_key = segs[2]
            part_type  = "model"
        elif len(segs) == 4 and segs[2] in PART_TYPES:
            # Single-figure design/fx: SideName.SideName.PartType.PartName
            figure_key = "default"
            part_type  = segs[2]
        else:
            # Single-figure model (3 segments) or fallback
            figure_key = "default"
            part_type  = "model"

        if figure_key not in figures:
            figures[figure_key] = {"model": [], "color": [], "design": [], "effect": []}
            figure_order.append(figure_key)

        bucket = {"Colors": "color", "Design": "design", "fx": "effect"}.get(part_type, "model")
        figures[figure_key][bucket].append(part)

    result_parts = list(bg_parts)

    for figure_key in figure_order:
        fig = figures[figure_key]
        file_key = side_name.lower() if figure_key == "default" else f"{side_name.lower()}.{figure_key.lower()}"
        model_name = side_name if figure_key == "default" else f"{side_name}.{figure_key}"

        if fig["model"]:
            result_parts.append(process_model_parts(model_name, fig["model"]))

        result_parts.extend(fig["color"])

        if fig["design"]:
            result_parts.append(_merge_parts_for_figure(file_key, fig["design"]))

        result_parts.extend(fig["effect"])

    mockup_info["parts"] = result_parts
    return mockup_info


def run_mockup_generation():
    """
    Standalone function to generate mockups from artwork.
    Can be run independently without going through the optimization process.
    """
    artwork_dir = os.path.join(ROOT_DIR, "optimized-mockup-infos/artworks")
    mockup_output_base_dir = os.path.join(ROOT_DIR, PREFIX, "mockups-output")
    
    # Create log file
    log_path = os.path.join(mockup_output_base_dir, "performance_log.txt")
    os.makedirs(mockup_output_base_dir, exist_ok=True)
    
    with open(log_path, 'w') as log_file:
        log_file.write("="*60 + "\n")
        log_file.write("MOCKUP GENERATION COMPARISON\n")
        log_file.write("="*60 + "\n")
        log_file.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write("="*60 + "\n")
        
        print("\n" + "="*60)
        print("MOCKUP GENERATION COMPARISON")
        print("="*60)
        
        # Load original mockup_infos
        original_path = os.path.join(f"{ROOT_DIR}/{PREFIX}", "mockup_infos.downloaded.json")
        with open(original_path, 'r') as f:
            original_mockup_infos = json.load(f)
        
        # Load optimized mockup_infos
        optimized_path = os.path.join(f"{ROOT_DIR}/{PREFIX}", "mockup_infos.optimized.json")
        with open(optimized_path, 'r') as f:
            optimized_mockup_infos = json.load(f)
        
        # Generate original mockups
        original_time, original_count, original_times, original_warp_count, orig_res = \
            generate_mockups_from_config(
                original_mockup_infos, 'original',
                artwork_dir, mockup_output_base_dir, log_file
            )

        # Generate optimized mockups (npy 1000x1000)
        optimized_time, optimized_count, optimized_times, optimized_warp_count, opt_res = \
            generate_mockups_from_config(
                optimized_mockup_infos, 'optimized',
                artwork_dir, mockup_output_base_dir, log_file
            )

        # Generate optimized mockups (json 100x100) — JS-renderer parity
        opt100_time, opt100_count, opt100_times, opt100_warp_count, opt100_res = \
            generate_mockups_from_config(
                optimized_mockup_infos, 'optimized_100x100',
                artwork_dir, mockup_output_base_dir, log_file,
                use_json_model=True
            )

        # Final comparison
        def _row(label, parts, t, count, res):
            cpu_total  = res['cpu_user_s'] + res['cpu_sys_s']
            efficiency = cpu_total / res['wall_s'] * 100 if res['wall_s'] > 0 else 0
            return (
                f"{label} ({parts} warp parts/view):\n"
                f"  Total time: {t:.3f}s  Avg: {t/count:.3f}s\n"
                f"  Peak memory:   {res['peak_mem_mb']:.1f} MB\n"
                f"  CPU time:      {res['cpu_user_s']:.3f}s user + {res['cpu_sys_s']:.3f}s sys\n"
                f"  Avg CPU usage: {res['avg_cpu_pct']:.1f}%  "
                f"(efficiency {efficiency:.1f}%)\n\n"
            )

        comparison_text  = f"\n{'='*60}\nFINAL COMPARISON\n{'='*60}\n"
        comparison_text += _row("Original",               original_warp_count,  original_time,  original_count,  orig_res)
        comparison_text += _row("Optimized npy 1000x1000", optimized_warp_count, optimized_time, optimized_count, opt_res)
        comparison_text += _row("Optimized json 100x100",  opt100_warp_count,    opt100_time,    opt100_count,    opt100_res)

        if original_time > 0:
            comparison_text += "Speedup vs original:\n"
            comparison_text += (f"  npy 1000x1000: {original_time/optimized_time:.2f}x"
                                f"  (parts reduced {original_warp_count-optimized_warp_count}/{original_warp_count})\n")
            comparison_text += f"  json 100x100:  {original_time/opt100_time:.2f}x\n"
            comparison_text += "Memory saving (peak):\n"
            comparison_text += f"  npy 1000x1000: {orig_res['peak_mem_mb'] - opt_res['peak_mem_mb']:+.1f} MB\n"
            comparison_text += f"  json 100x100:  {orig_res['peak_mem_mb'] - opt100_res['peak_mem_mb']:+.1f} MB\n"

        comparison_text += "=" * 60 + "\n"
        print(comparison_text)
        log_file.write(comparison_text)
        print(f"\n✓ Performance log saved to: {log_path}")

    # Generate comparison chart
    chart_path = os.path.join(mockup_output_base_dir, "performance_comparison.png")
    generate_comparison_chart(original_time, optimized_time, original_times, optimized_times,
                             original_warp_count, optimized_warp_count, chart_path)


async def main():
    with open(f"{ROOT_DIR}/{PREFIX}/mockup_infos.json", 'r') as f:
        mockup_infos = json.load(f)
    
    downloaded_mockup_infos_path = await download_mockup_infos(mockup_infos)
    with open(downloaded_mockup_infos_path, 'r') as f:
        mockup_infos = json.load(f)

    for mockup_info in mockup_infos['mockup_infos']:
        mockup_info = process_for_side(mockup_info)
    
    output_path = os.path.join(f"{ROOT_DIR}/{PREFIX}", "mockup_infos.optimized.json")
    with open(output_path, 'w') as f:
        json.dump(mockup_infos, f, indent=2)
    
    web_mockup_infos = json.dumps(mockup_infos, indent=2)
    web_mockup_infos = web_mockup_infos.replace(ROOT_DIR, ".")
    output_path = os.path.join(f"{ROOT_DIR}/{PREFIX}", "mockup_infos.optimized_web.json")
    with open(output_path, 'w') as f:
        f.write(web_mockup_infos)
    
    
    run_mockup_generation()


if __name__ == "__main__":
    asyncio.run(main())