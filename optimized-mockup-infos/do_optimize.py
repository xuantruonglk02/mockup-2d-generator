import json
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
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from warp_image.tps import numpy as tps


ROOT_DIR = "/home/dev/code/test-color-mockup-2d"
PREFIX = "optimized-mockup-infos/mockups/crop_top_baseball_jersey_without_piping"


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


def join_model_data(base, model, mask_path):
    mask_img = Image.open(mask_path).convert('RGBA')
    mask_array = np.array(mask_img)
    alpha_channel = mask_array[:, :, 3]
    has_alpha = alpha_channel > 0
    base[has_alpha] = model[has_alpha]


def resize_model(model_data, mask, target_size):
    """
    Resize với soft edge để tránh jagged artifacts
    """
    from scipy.ndimage import distance_transform_edt, gaussian_filter
    
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


def process_design_parts(side_name, parts):
    original_size = (1000, 1000)
    base_model = np.full((*original_size, 2), np.nan, dtype=np.float32)
    base_mask = Image.new('RGBA', original_size, (0, 0, 0, 0))
    
    for part in parts:
        model_path = part["warp_info"]["model"]
        mask_path = part["mask_path"]
        
        model_data = np.load(model_path, allow_pickle=True)
        print(f"Original model shape: {model_data.shape}")
        
        join_model_data(base_model, model_data, mask_path)
        
        img = Image.open(mask_path).convert('RGBA')
        base_mask.paste(img, (0, 0), img)
    
    base_mask_array = np.array(base_mask)
    alpha_channel = base_mask_array[:, :, 3]
    mask = alpha_channel > 0
    
    # base_model = resize_model(base_model, mask, (100, 100))
    
    model_path = f"{ROOT_DIR}/{PREFIX}/optimized/{side_name.lower()}.warp.npy"
    np.save(model_path, base_model)
    
    mask_path = f"{ROOT_DIR}/{PREFIX}/optimized/{side_name.lower()}.warp_mask.png"
    base_mask.save(mask_path)
    
    json_path = f"{ROOT_DIR}/{PREFIX}/optimized/{side_name.lower()}.warp.json"
    with open(json_path, 'w') as f:
        json.dump(base_model.tolist(), f, separators=(",", ":"))
    
    return {
        "name": f"{side_name}.Design",
        "side": "front",
        # "mask_path": mask_path,
        "warp_type": "warp_npy",
        "warp_info": {
            "model": model_path,
            "model_json": json_path,
            "artwork_width": 3000,
            "artwork_height": 3000,
        },
        "effects": [],
        "fill": 100,
        "opacity": 100
    }


def warp_image(image, model_path, artwork_size, warped_size):
    """
    Warp image using TPS (Thin Plate Spline) transformation
    
    Args:
        image: numpy array or PIL Image
        model_path: path to the .npy model file
        artwork_size: tuple (width, height) for artwork
        warped_size: tuple (width, height) for final warped image
    
    Returns:
        PIL Image of warped result
    """
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Resize the image with high-quality interpolation
    resized_img = cv2.resize(image, artwork_size, cv2.INTER_LANCZOS4)
    
    # Load the transformation grid
    grid = np.load(model_path, allow_pickle=True)
    
    # Generate the remapping coordinates
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


def generate_mockup(mockup_info, artwork_path, output_path, mask_mode=False):
    """
    Generate a mockup image from mockup_info and artwork
    
    Args:
        mockup_info: dict containing mockup configuration
        artwork_path: path to artwork image
        output_path: path to save the generated mockup
        mask_mode: if True, apply mask to warped design parts
    """
    size = mockup_info.get('size', {'width': 1000, 'height': 1000})
    canvas_size = (size['width'], size['height'])
    
    # Create base canvas
    canvas = Image.new('RGBA', canvas_size, (0, 0, 0, 0))
    
    # Load artwork once
    artwork = Image.open(artwork_path).convert('RGBA')
    
    for part in mockup_info['parts']:
        part_name = part['name']
        
        # Handle background and model parts
        if 'image_path' in part and part.get('warp_type') != 'warp_npy':
            layer = Image.open(part['image_path']).convert('RGBA')
            
            # Apply blend mode if specified
            if 'blend_mode' in part and part['blend_mode'] != 'normal':
                canvas_array = np.array(canvas)
                layer_array = np.array(layer)
                blended = apply_blend_mode(canvas_array, layer_array, part['blend_mode'])
                canvas = Image.fromarray(blended, 'RGBA')
            else:
                canvas.paste(layer, (0, 0), layer)
        
        # Handle design/warp parts
        elif part.get('warp_type') == 'warp_npy':
            warp_info = part['warp_info']
            model_path = warp_info['model']
            artwork_size = (warp_info['artwork_width'], warp_info['artwork_height'])
            
            # Warp the artwork
            warped = warp_image(artwork, model_path, artwork_size, canvas_size)
            
            # Apply mask if specified and in mask mode (original version)
            if mask_mode and 'mask_path' in part:
                mask = Image.open(part['mask_path']).convert('RGBA')
                warped_array = np.array(warped)
                mask_array = np.array(mask)
                
                # Use mask alpha as the design alpha
                warped_array[:, :, 3] = np.minimum(warped_array[:, :, 3], mask_array[:, :, 3])
                warped = Image.fromarray(warped_array, 'RGBA')
            
            # Apply opacity if specified
            opacity = part.get('opacity', 100)
            if opacity < 100:
                warped_array = np.array(warped)
                warped_array[:, :, 3] = (warped_array[:, :, 3] * opacity / 100).astype(np.uint8)
                warped = Image.fromarray(warped_array, 'RGBA')
            
            # Composite onto canvas
            canvas.paste(warped, (0, 0), warped)
    
    # Save the result
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    canvas.save(output_path)


def generate_mockups_from_config(mockup_infos, config_type, artwork_dir, output_base_dir, log_file):
    """
    Generate mockups from a specific configuration (original or optimized)
    
    Args:
        mockup_infos: dict containing mockup configuration
        config_type: str - 'original' or 'optimized'
        artwork_dir: path to artwork directory
        output_base_dir: base output directory
        log_file: file handle for logging
    
    Returns:
        tuple: (total_time, mockup_count, individual_times)
    """
    msg = f"\n{'='*60}\nGenerating {config_type.upper()} mockups\n{'='*60}"
    print(msg)
    log_file.write(msg + "\n")
    
    mockup_output_dir = os.path.join(output_base_dir, config_type)
    
    # Get all artwork files
    artwork_files = [f for f in os.listdir(artwork_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    if not artwork_files:
        msg = f"No artwork files found in {artwork_dir}"
        print(msg)
        log_file.write(msg + "\n")
        return 0, 0, []
    
    total_start_time = time.time()
    mockup_count = 0
    individual_times = []
    
    # mask_mode is True for original (uses individual masks), False for optimized
    mask_mode = (config_type == 'original')
    
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
            
            generate_mockup(mockup_info, artwork_path, output_path, mask_mode)
            
            mockup_duration = time.time() - mockup_start
            individual_times.append(mockup_duration)
            msg = f"✓ ({mockup_duration:.3f}s)"
            print(msg)
            log_file.write(" " + msg + "\n")
            mockup_count += 1
    
    total_time = time.time() - total_start_time
    
    msg = f"\n{'='*60}\n{config_type.upper()} Summary:\n  Total mockups: {mockup_count}\n  Total time: {total_time:.3f}s\n  Average time per mockup: {total_time/mockup_count:.3f}s\n  Output directory: {mockup_output_dir}\n{'='*60}"
    print(msg)
    log_file.write(msg + "\n")
    
    return total_time, mockup_count, individual_times


def generate_comparison_chart(original_time, optimized_time, original_times, optimized_times, output_path):
    """
    Generate a comparison chart showing performance metrics
    
    Args:
        original_time: total time for original
        optimized_time: total time for optimized
        original_times: list of individual mockup times for original
        optimized_times: list of individual mockup times for optimized
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
    
    Optimized:
      • Total Time: {optimized_time:.3f}s
      • Average/Mockup: {avg_optimized:.3f}s
      • Mockups: {len(optimized_times)}
    
    Improvement:
      • Speedup: {speedup:.2f}x faster
      • Time Saved: {time_saved:.3f}s
      • Reduction: {percent_saved:.1f}%
    """
    
    ax4.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round',
             facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Comparison chart saved to: {output_path}")


def process_for_side(mockup_info):
    side_name = mockup_info["name"]

    bg_parts = []
    model_parts = []
    color_parts = []
    design_parts = []
    effect_parts = []
    
    for index, part in enumerate(mockup_info["parts"]):
        part_name_splitted = part["name"].split(".")
        
        if part_name_splitted[1] == "BG":
            bg_parts.append(part)
        elif part_name_splitted[2] == "Colors" or (len(part_name_splitted) == 5 and part_name_splitted[3] == "Colors"):
            color_parts.append(part)
        elif part_name_splitted[2] == "Design" or (len(part_name_splitted) == 5 and part_name_splitted[3] == "Design"):
            design_parts.append(part)
        elif part_name_splitted[2] == "fx" or (len(part_name_splitted) == 5 and part_name_splitted[3] == "fx"):
            effect_parts.append(part)
        else:
            model_parts.append(part)
            
    dir = f"{ROOT_DIR}/{PREFIX}/optimized"
    if not os.path.exists(dir):
        os.mkdir(dir)
    
    model_part = process_model_parts(side_name, model_parts)
    design_part = process_design_parts(side_name, design_parts)
    
    mockup_info["parts"] = [
        *bg_parts,
        model_part,
        *color_parts,
        design_part,
        *effect_parts,
    ]
    
    return mockup_info


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
    
    # Generate mockups from artwork - both original and optimized
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
        
        # Generate original mockups
        original_time, original_count, original_times = generate_mockups_from_config(
            original_mockup_infos,
            'original',
            artwork_dir,
            mockup_output_base_dir,
            log_file
        )
        
        # Generate optimized mockups
        optimized_time, optimized_count, optimized_times = generate_mockups_from_config(
            mockup_infos,
            'optimized',
            artwork_dir,
            mockup_output_base_dir,
            log_file
        )
        
        # Final comparison
        comparison_text = f"\n{'='*60}\nFINAL COMPARISON\n{'='*60}\n"
        comparison_text += f"Original:\n  Mockups: {original_count}\n  Total time: {original_time:.3f}s\n"
        comparison_text += f"  Avg time/mockup: {original_time/original_count:.3f}s\n\n"
        comparison_text += f"Optimized:\n  Mockups: {optimized_count}\n  Total time: {optimized_time:.3f}s\n"
        comparison_text += f"  Avg time/mockup: {optimized_time/optimized_count:.3f}s\n\n"
        
        if original_time > 0:
            speedup = original_time / optimized_time
            time_saved = original_time - optimized_time
            comparison_text += f"Performance Improvement:\n"
            comparison_text += f"  Speedup: {speedup:.2f}x faster\n"
            comparison_text += f"  Time saved: {time_saved:.3f}s ({time_saved/original_time*100:.1f}%)\n"
        
        comparison_text += "="*60 + "\n"
        
        print(comparison_text)
        log_file.write(comparison_text)
        
        print(f"\n✓ Performance log saved to: {log_path}")
    
    # Generate comparison chart
    chart_path = os.path.join(mockup_output_base_dir, "performance_comparison.png")
    generate_comparison_chart(original_time, optimized_time, original_times, optimized_times, chart_path)
    

if __name__ == "__main__":
    asyncio.run(main())