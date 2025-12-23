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
    

if __name__ == "__main__":
    asyncio.run(main())