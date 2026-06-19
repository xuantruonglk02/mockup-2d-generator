import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def visualize_mask_and_grid(mask_path, grid_path, output_path="debug_viz.png"):
    """
    Visualize alpha edge của mask và valid region của grid
    để hiểu chính xác vùng semi-transparent overlap với NaN grid như thế nào.
    """
    # Load mask
    mask = Image.open(mask_path).convert('RGBA')
    mask_arr = np.array(mask)
    alpha = mask_arr[:, :, 3]  # 0-255
    
    # Load grid
    grid = np.load(grid_path, allow_pickle=True)  # HxWx2
    
    # Grid valid region: cả 2 channel không phải NaN
    grid_valid = ~(np.isnan(grid[:, :, 0]) | np.isnan(grid[:, :, 1]))
    
    # Resize grid valid mask về cùng size với mask image
    import cv2
    grid_valid_resized = cv2.resize(
        grid_valid.astype(np.uint8) * 255,
        (mask_arr.shape[1], mask_arr.shape[0]),
        interpolation=cv2.INTER_NEAREST
    ) > 0
    
    # Vùng semi-transparent: alpha > 0 và alpha < 255
    semi_transparent = (alpha > 0) & (alpha < 255)
    
    # Vùng nguy hiểm: semi-transparent nhưng grid là NaN
    danger_zone = semi_transparent & ~grid_valid_resized
    
    # Vùng alpha > 0 nhưng grid NaN
    alpha_present_grid_nan = (alpha > 0) & ~grid_valid_resized
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Alpha channel
    ax = axes[0, 0]
    im = ax.imshow(alpha, cmap='gray', vmin=0, vmax=255)
    ax.set_title(f'Mask Alpha Channel\n(min={alpha.min()}, max={alpha.max()})')
    plt.colorbar(im, ax=ax)
    
    # 2. Semi-transparent zone
    ax = axes[0, 1]
    semi_viz = np.zeros((*alpha.shape, 3), dtype=np.uint8)
    semi_viz[alpha == 255] = [0, 255, 0]      # Full opaque: green
    semi_viz[semi_transparent] = [255, 165, 0] # Semi: orange
    semi_viz[alpha == 0] = [50, 50, 50]        # Transparent: dark gray
    ax.imshow(semi_viz)
    ax.set_title(f'Alpha Zones\nGreen=opaque, Orange=semi({semi_transparent.sum()}px), Gray=transparent')
    
    # 3. Grid valid region
    ax = axes[0, 2]
    grid_viz = np.zeros((*alpha.shape, 3), dtype=np.uint8)
    grid_viz[grid_valid_resized] = [0, 200, 0]
    grid_viz[~grid_valid_resized] = [200, 0, 0]
    ax.imshow(grid_viz)
    ax.set_title('Grid Valid Region\nGreen=valid, Red=NaN')
    
    # 4. Danger zone: semi-transparent + grid NaN
    ax = axes[1, 0]
    danger_viz = np.zeros((*alpha.shape, 3), dtype=np.uint8)
    danger_viz[grid_valid_resized & (alpha > 0)] = [0, 200, 0]   # safe
    danger_viz[danger_zone] = [255, 0, 0]                         # danger: semi + NaN
    danger_viz[alpha_present_grid_nan & ~semi_transparent] = [255, 128, 0]  # opaque + NaN
    ax.imshow(danger_viz)
    n_danger = danger_zone.sum()
    ax.set_title(f'Danger Zone (semi-transparent + NaN grid)\nRed={n_danger}px')
    
    # 5. Alpha histogram tại viền
    ax = axes[1, 1]
    # Lấy pixel viền: dilate opaque - opaque
    import cv2 as cv
    opaque = (alpha == 255).astype(np.uint8)
    dilated = cv.dilate(opaque, np.ones((5,5), np.uint8), iterations=3)
    border_region = (dilated > 0) & (alpha < 255) & (alpha > 0)
    
    if border_region.sum() > 0:
        border_alphas = alpha[border_region]
        ax.hist(border_alphas, bins=50, color='orange', edgecolor='black')
        ax.set_xlabel('Alpha value (0-255)')
        ax.set_ylabel('Pixel count')
        ax.set_title(f'Alpha distribution at border\n({border_region.sum()} border pixels)')
    else:
        ax.text(0.5, 0.5, 'No border pixels found', ha='center', va='center')
        ax.set_title('Alpha distribution at border')
    
    # 6. Overlap: alpha edge vs grid boundary
    ax = axes[1, 2]
    overlap_viz = np.zeros((*alpha.shape, 3), dtype=np.uint8)
    # Grid boundary (pixels at edge of valid region)
    grid_boundary = cv.dilate(grid_valid_resized.astype(np.uint8), np.ones((3,3), np.uint8)) \
                    .astype(bool) & ~grid_valid_resized
    
    overlap_viz[alpha > 0] = [100, 100, 255]      # alpha present: blue
    overlap_viz[grid_boundary] = [255, 255, 0]     # grid boundary: yellow
    overlap_viz[alpha > 0 & grid_boundary] = [255, 0, 255]  # overlap: magenta
    ax.imshow(overlap_viz)
    ax.set_title('Blue=alpha>0, Yellow=grid boundary\nMagenta=overlap (problem area)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"MASK: {mask_path}")
    print(f"GRID: {grid_path}")
    print(f"{'='*50}")
    print(f"Mask size: {mask_arr.shape[:2]}")
    print(f"Grid size: {grid.shape[:2]}")
    print(f"Alpha stats:")
    print(f"  Fully opaque (255): {(alpha==255).sum()} px")
    print(f"  Semi-transparent (1-254): {semi_transparent.sum()} px")
    print(f"  Transparent (0): {(alpha==0).sum()} px")
    print(f"Grid stats:")
    print(f"  Valid (non-NaN): {grid_valid.sum()} px")
    print(f"  NaN: {(~grid_valid).sum()} px")
    print(f"Danger zones:")
    print(f"  Semi-transparent + NaN grid: {danger_zone.sum()} px  ← CAUSE OF JAGGED")
    print(f"  Alpha>0 + NaN grid: {alpha_present_grid_nan.sum()} px")
    print(f"{'='*50}")
    print(f"Saved: {output_path}")
    
    return {
        'danger_pixels': danger_zone.sum(),
        'semi_transparent_pixels': semi_transparent.sum(),
        'grid_valid_pixels': grid_valid.sum(),
    }


# Chạy với một mảnh cụ thể (ví dụ tay trái)
result = visualize_mask_and_grid(
    mask_path="/home/dev/code/test-color-mockup-2d/optimized-mockup-infos/mockups/crop_top_baseball_jersey_without_piping/downloads/product-mockups/69451bd90779d92084f31a2d/image-mask/Back.Back.Design.Back_2.png",
    grid_path="/home/dev/code/test-color-mockup-2d/optimized-mockup-infos/mockups/crop_top_baseball_jersey_without_piping/downloads/product-mockups/69451bd90779d92084f31a2d/model/Back.Back.Design.Back_2.model.npy",
    output_path="debug_mask_grid.png"
)