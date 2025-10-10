import json
import numpy as np
from PIL import Image
import cv2

ROOT_DIR = "./optimized-mockup-infos/18000bus"

with open(f"{ROOT_DIR}/mockup_infos.json", 'r') as f:
    mockup_infos = json.load(f)

TARGET_SIZE = (100, 100)  # Resize to 100x100

def process_for_side(mockup_info):
    side_name = mockup_info["name"]

    npy_files = []
    for part in mockup_info['parts']:
        if 'warp_info' in part:
            warp_info = part['warp_info']
            npy_path = warp_info.get('model')
            if npy_path:
                npy_files.append({
                    "model": npy_path,
                    "mask": part["mask_path"]
                })

    # Initialize merged arrays with target size
    optimized_npy_data = np.full((TARGET_SIZE[0], TARGET_SIZE[1], 2), -10, dtype=np.float32)
    optimized_mask = Image.new('RGBA', (1000,1000), (0, 0, 0, 0))

    for npy_file in npy_files:
        model_path = npy_file["model"]
        mask_path = npy_file["mask"]
        
        model_data = np.load(model_path, allow_pickle=True)
        print(f"Original model shape: {model_data.shape}")
        
        mask_img = Image.open(mask_path).convert('RGBA')
        mask_array = np.array(mask_img)
        alpha_channel = mask_array[:, :, 3]
        
        # Resize model_data directly (no -10 influence since not present initially)
        resized_temp = cv2.resize(model_data, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)
        
        # Resize alpha for cropping (use INTER_AREA to preserve coverage)
        resized_alpha = cv2.resize(alpha_channel, TARGET_SIZE, interpolation=cv2.INTER_AREA)
        
        kernel = np.ones((3, 3), np.uint8)
        dilated_alpha = cv2.dilate((resized_alpha > 0).astype(np.uint8), kernel, iterations=1)
        
        # Crop: set -10 where resized_alpha low
        resized_has_data = dilated_alpha.astype(bool)
        resized_npy_data = np.full_like(resized_temp, -10)  # Initialize with -10
        resized_npy_data[resized_has_data] = resized_temp[resized_has_data]
        
        print(f"Resized model shape: {resized_npy_data.shape}")
        
        # Merge into optimized
        optimized_has_data = resized_npy_data[:, :, 0] != -10
        optimized_npy_data[optimized_has_data] = resized_npy_data[optimized_has_data]
        
        # Paste original mask (no resize) to avoid blur for warping
        optimized_mask.paste(mask_img, (0, 0), mask_img)
        
        print("-" * 80)
    
    # Save merged data (no additional resizing)
    np.save(f"{ROOT_DIR}/optimized_npy_data.{side_name.lower()}.npy", optimized_npy_data)
    optimized_mask.save(f"{ROOT_DIR}/optimized_mask.{side_name.lower()}.png")
    
    # Save as JSON
    with open(f"{ROOT_DIR}/optimized_npy_data.{side_name.lower()}.json", 'w') as f:
        json.dump(optimized_npy_data.tolist(), f, separators=(",", ":"))
    
    print(f"✓ Saved {side_name}:")
    print(f"  - optimized_npy_data.{side_name.lower()}.npy (shape: {optimized_npy_data.shape})")
    print(f"  - optimized_mask.{side_name.lower()}.png")
    print(f"  - optimized_npy_data.{side_name.lower()}.json\n")

for mockup_info in mockup_infos['mockup_infos']:
    process_for_side(mockup_info)