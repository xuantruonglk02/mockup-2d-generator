"""
Script khám phá data: masks và warp models
Mục tiêu: hiểu rõ
  1. Masks có overlap nhau tại seam giữa các parts không?
  2. Alpha tại seam là binary hay gradient?
  3. Warp coordinates có liên tục tại seam không?
  4. Từ đó kết luận: có thể merge an toàn không?
"""
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

ROOT_DIR = "/home/dev/code/test-color-mockup-2d"
PREFIX = "optimized-mockup-infos/mockups/crop_top_baseball_jersey_without_piping"
OUT_DIR = f"{ROOT_DIR}/{PREFIX}/exploration"
os.makedirs(OUT_DIR, exist_ok=True)

DOWNLOADED_JSON = f"{ROOT_DIR}/{PREFIX}/mockup_infos.downloaded.json"


# ── helpers ──────────────────────────────────────────────────────────────────

def load_alpha(path: str) -> np.ndarray:
    """Load mask file, return alpha channel as float32 [0,1], shape (H,W)."""
    img = Image.open(path).convert("RGBA")
    return np.array(img)[:, :, 3].astype(np.float32) / 255.0


def get_design_parts(mockup_info: dict) -> list:
    return [p for p in mockup_info["parts"] if p.get("warp_type") == "warp_npy"]


# ── 1. Overview: visualise tất cả masks của mỗi side ─────────────────────────

def visualise_masks_per_side(mockup_infos: list):
    for view in mockup_infos:
        name = view["name"]
        parts = get_design_parts(view)
        if not parts:
            continue

        n = len(parts)
        fig, axes = plt.subplots(1, n + 1, figsize=(4 * (n + 1), 4))
        fig.suptitle(f"[{name}] Individual masks ({n} design parts)", fontsize=13)

        alphas = []
        for i, part in enumerate(parts):
            alpha = load_alpha(part["mask_path"])
            alphas.append(alpha)
            ax = axes[i] if n > 0 else axes
            ax.imshow(alpha, cmap="hot", vmin=0, vmax=1)
            label = part["name"].split(".")[-1]
            ax.set_title(f"{label}\nmax={alpha.max():.2f}", fontsize=9)
            ax.axis("off")

        # Merged mask (alpha composite)
        merged = np.zeros_like(alphas[0])
        for a in alphas:
            merged = merged + a * (1 - merged)

        axes[-1].imshow(merged, cmap="hot", vmin=0, vmax=1)
        axes[-1].set_title(f"Merged\n(alpha-composite)", fontsize=9)
        axes[-1].axis("off")

        fig.tight_layout()
        out = f"{OUT_DIR}/{name}_masks_overview.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"Saved: {out}")


# ── 2. Overlap analysis: pairwise giữa các masks ────────────────────────────

def analyse_overlap(mockup_infos: list):
    report = {}

    for view in mockup_infos:
        name = view["name"]
        parts = get_design_parts(view)
        if len(parts) < 2:
            continue

        alphas = [(p["name"].split(".")[-1], load_alpha(p["mask_path"])) for p in parts]
        n = len(alphas)

        print(f"\n{'='*60}")
        print(f"[{name}] Overlap analysis ({n} parts)")
        print(f"{'='*60}")

        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                name_i, a_i = alphas[i]
                name_j, a_j = alphas[j]

                # Vùng cả 2 đều có alpha > 0
                both_nonzero = (a_i > 0) & (a_j > 0)
                overlap_pixels = both_nonzero.sum()

                # Trong vùng overlap, alpha có significant value không?
                both_significant = (a_i > 0.05) & (a_j > 0.05)
                significant_pixels = both_significant.sum()

                # Tại vùng overlap, alpha sum > 1 không? (double coverage)
                sum_alpha = a_i + a_j
                double_coverage = (sum_alpha > 1.0).sum()

                print(f"  {name_i} × {name_j}:")
                print(f"    Pixels cả 2 non-zero:   {overlap_pixels:6d}")
                print(f"    Pixels cả 2 > 5%:       {significant_pixels:6d}")
                print(f"    Pixels sum alpha > 1.0: {double_coverage:6d}")

                pairs.append({
                    "i": name_i, "j": name_j,
                    "overlap": overlap_pixels,
                    "significant": significant_pixels,
                    "double_coverage": double_coverage,
                    "alpha_i": a_i,
                    "alpha_j": a_j,
                })

        report[name] = pairs

    return report


# ── 3. Seam zone deep-dive: alpha distribution tại biên tiếp giáp ───────────

def analyse_seam_zone(mockup_infos: list):
    for view in mockup_infos:
        name = view["name"]
        parts = get_design_parts(view)
        if len(parts) < 2:
            continue

        alphas = [(p["name"].split(".")[-1], load_alpha(p["mask_path"])) for p in parts]
        n = len(alphas)

        for i in range(n):
            for j in range(i + 1, n):
                label_i, a_i = alphas[i]
                label_j, a_j = alphas[j]

                # Seam zone: nơi ít nhất một mask có alpha ở dải (0.01, 0.99)
                seam_i = (a_i > 0.01) & (a_i < 0.99)
                seam_j = (a_j > 0.01) & (a_j < 0.99)
                seam_union = seam_i | seam_j

                if seam_union.sum() < 10:
                    print(f"  [{name}] {label_i}×{label_j}: Rất ít seam pixels → masks có vẻ binary tại biên")
                    continue

                fig, axes = plt.subplots(2, 3, figsize=(14, 8))
                fig.suptitle(f"[{name}] Seam analysis: {label_i} vs {label_j}", fontsize=12)

                # Panel 1: alpha_i
                axes[0, 0].imshow(a_i, cmap="hot", vmin=0, vmax=1)
                axes[0, 0].set_title(f"Alpha: {label_i}", fontsize=9)
                axes[0, 0].axis("off")

                # Panel 2: alpha_j
                axes[0, 1].imshow(a_j, cmap="hot", vmin=0, vmax=1)
                axes[0, 1].set_title(f"Alpha: {label_j}", fontsize=9)
                axes[0, 1].axis("off")

                # Panel 3: overlap heatmap (min của 2 alpha)
                overlap_map = np.minimum(a_i, a_j)
                axes[0, 2].imshow(overlap_map, cmap="plasma", vmin=0, vmax=1)
                axes[0, 2].set_title(f"Overlap (min alpha)", fontsize=9)
                axes[0, 2].axis("off")

                # Panel 4: seam zone mask
                seam_display = np.zeros((*a_i.shape, 3))
                seam_display[seam_i] = [1, 0.3, 0.3]  # đỏ = seam của i
                seam_display[seam_j] = [0.3, 0.3, 1]  # xanh = seam của j
                seam_display[seam_i & seam_j] = [1, 0, 1]  # tím = overlap
                axes[1, 0].imshow(seam_display)
                axes[1, 0].set_title(
                    f"Seam zones\nRed={label_i}|Blue={label_j}|Purple=both\n"
                    f"seam_i={seam_i.sum()} seam_j={seam_j.sum()} both={( seam_i & seam_j).sum()}",
                    fontsize=8
                )
                axes[1, 0].axis("off")

                # Panel 5: alpha histogram tại seam
                ax = axes[1, 1]
                ax.hist(a_i[seam_i].ravel(), bins=50, alpha=0.6, color="red", label=label_i, density=True)
                ax.hist(a_j[seam_j].ravel(), bins=50, alpha=0.6, color="blue", label=label_j, density=True)
                ax.set_xlabel("Alpha value")
                ax.set_ylabel("Density")
                ax.set_title("Alpha histogram at seam pixels")
                ax.legend(fontsize=8)

                # Panel 6: sum alpha tại overlap zone
                ax = axes[1, 2]
                sum_alpha = a_i + a_j
                both_mask = (a_i > 0.01) & (a_j > 0.01)
                if both_mask.sum() > 0:
                    ax.hist(sum_alpha[both_mask].ravel(), bins=50, color="purple", alpha=0.7)
                    ax.axvline(1.0, color="red", linestyle="--", label="sum=1.0")
                    ax.set_xlabel("Sum of alpha (a_i + a_j)")
                    ax.set_title(f"Sum alpha where both > 1%\n({both_mask.sum()} pixels)")
                    ax.legend(fontsize=8)
                else:
                    ax.text(0.5, 0.5, "No overlap", ha="center", va="center")
                    ax.set_title("No overlap region")

                fig.tight_layout()
                safe_label_i = label_i.replace(" ", "_")
                safe_label_j = label_j.replace(" ", "_")
                out = f"{OUT_DIR}/{name}_seam_{safe_label_i}_vs_{safe_label_j}.png"
                fig.savefig(out, dpi=110)
                plt.close(fig)
                print(f"Saved: {out}")


# ── 4. Warp coordinate field visualisation ───────────────────────────────────

def visualise_warp_fields(mockup_infos: list):
    for view in mockup_infos:
        name = view["name"]
        parts = get_design_parts(view)
        if not parts:
            continue

        n = len(parts)
        fig, axes = plt.subplots(2, n, figsize=(5 * n, 9))
        if n == 1:
            axes = axes.reshape(2, 1)
        fig.suptitle(f"[{name}] Warp coordinate fields", fontsize=12)

        for i, part in enumerate(parts):
            label = part["name"].split(".")[-1]
            model_path = part["warp_info"]["model"]
            grid = np.load(model_path, allow_pickle=True)  # (H, W, 2)

            alpha = load_alpha(part["mask_path"])
            valid = ~np.isnan(grid[:, :, 0])

            # X coords
            x_ch = grid[:, :, 0].copy()
            x_ch[~valid] = np.nan
            im0 = axes[0, i].imshow(x_ch, cmap="RdBu", interpolation="nearest")
            axes[0, i].set_title(f"{label}\nX coord (→artwork col)", fontsize=8)
            axes[0, i].axis("off")
            plt.colorbar(im0, ax=axes[0, i], fraction=0.046)

            # Y coords
            y_ch = grid[:, :, 1].copy()
            y_ch[~valid] = np.nan
            im1 = axes[1, i].imshow(y_ch, cmap="RdBu", interpolation="nearest")
            axes[1, i].set_title(f"{label}\nY coord (→artwork row)", fontsize=8)
            axes[1, i].axis("off")
            plt.colorbar(im1, ax=axes[1, i], fraction=0.046)

            print(f"  [{name}] {label}: grid shape={grid.shape}, valid={valid.sum()}, "
                  f"X=[{np.nanmin(x_ch):.1f},{np.nanmax(x_ch):.1f}] "
                  f"Y=[{np.nanmin(y_ch):.1f},{np.nanmax(y_ch):.1f}]")

        fig.tight_layout()
        out = f"{OUT_DIR}/{name}_warp_fields.png"
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print(f"Saved: {out}")


# ── 5. Warp discontinuity at seams ──────────────────────────────────────────

def analyse_warp_discontinuity(mockup_infos: list):
    """
    Tại vùng biên (seam) giữa 2 parts, warp coordinates của 2 parts
    có gần nhau không? Nếu không → merge sẽ tạo artifacts.
    """
    for view in mockup_infos:
        name = view["name"]
        parts = get_design_parts(view)
        if len(parts) < 2:
            continue

        loaded = []
        for p in parts:
            grid = np.load(p["warp_info"]["model"], allow_pickle=True)
            alpha = load_alpha(p["mask_path"])
            label = p["name"].split(".")[-1]
            loaded.append((label, grid, alpha))

        n = len(loaded)
        for i in range(n):
            for j in range(i + 1, n):
                label_i, grid_i, alpha_i = loaded[i]
                label_j, grid_j, alpha_j = loaded[j]

                # Seam zone: nơi cả 2 có alpha > 0.05
                seam = (alpha_i > 0.05) & (alpha_j > 0.05)
                if seam.sum() < 5:
                    # Thử biên của từng mask (non-zero adjacency)
                    # Mở rộng mask i 5px, giao với mask j > 0
                    import cv2
                    kernel = np.ones((11, 11), np.uint8)
                    dilated_i = cv2.dilate((alpha_i > 0.05).astype(np.uint8), kernel) > 0
                    dilated_j = cv2.dilate((alpha_j > 0.05).astype(np.uint8), kernel) > 0
                    seam = (dilated_i & (alpha_j > 0.05)) | (dilated_j & (alpha_i > 0.05))

                if seam.sum() < 5:
                    print(f"  [{name}] {label_i}×{label_j}: không có seam zone → masks không tiếp giáp")
                    continue

                valid_i = ~np.isnan(grid_i[:, :, 0])
                valid_j = ~np.isnan(grid_j[:, :, 0])
                sample_zone = seam & valid_i & valid_j

                if sample_zone.sum() < 5:
                    print(f"  [{name}] {label_i}×{label_j}: seam zone nhưng thiếu valid coords ở một trong 2 model")
                    continue

                dx = grid_i[:, :, 0][sample_zone] - grid_j[:, :, 0][sample_zone]
                dy = grid_i[:, :, 1][sample_zone] - grid_j[:, :, 1][sample_zone]
                dist = np.sqrt(dx**2 + dy**2)

                print(f"\n  [{name}] {label_i} × {label_j} — warp discontinuity tại seam:")
                print(f"    Số pixels seam có cả 2 valid: {sample_zone.sum()}")
                print(f"    |coord_i - coord_j|: mean={dist.mean():.2f}  median={np.median(dist):.2f}  max={dist.max():.2f} px")
                print(f"    → {'⚠️  DISCONTINUOUS (artifacts nếu merge)' if np.median(dist) > 5 else '✅ Continuous (safe to merge)'}")

                # Visualise
                fig, axes = plt.subplots(1, 3, figsize=(14, 4))
                fig.suptitle(
                    f"[{name}] Warp discontinuity: {label_i} vs {label_j}\n"
                    f"median diff = {np.median(dist):.2f}px  max = {dist.max():.2f}px",
                    fontsize=11
                )

                # Diff map
                diff_map = np.full(alpha_i.shape, np.nan)
                diff_map[sample_zone] = dist
                im = axes[0].imshow(diff_map, cmap="hot", vmin=0)
                axes[0].set_title("Coord distance |coord_i - coord_j| at seam")
                axes[0].axis("off")
                plt.colorbar(im, ax=axes[0], fraction=0.046)

                # Histogram
                axes[1].hist(dist, bins=50, color="orangered", alpha=0.8)
                axes[1].axvline(np.median(dist), color="blue", linestyle="--", label=f"median={np.median(dist):.1f}")
                axes[1].set_xlabel("Pixel distance between coords")
                axes[1].set_title("Distribution of coord differences")
                axes[1].legend()

                # Alpha overlay at seam
                overlay = np.zeros((*alpha_i.shape, 3))
                overlay[:, :, 0] = alpha_i
                overlay[:, :, 2] = alpha_j
                seam_highlight = seam.astype(np.float32)
                overlay[:, :, 1] = seam_highlight * 0.8
                axes[2].imshow(np.clip(overlay, 0, 1))
                axes[2].set_title(f"Red={label_i}  Blue={label_j}  Green=seam zone")
                axes[2].axis("off")

                fig.tight_layout()
                safe_i = label_i.replace(" ", "_")
                safe_j = label_j.replace(" ", "_")
                out = f"{OUT_DIR}/{name}_warp_discontinuity_{safe_i}_vs_{safe_j}.png"
                fig.savefig(out, dpi=110)
                plt.close(fig)
                print(f"    Saved: {out}")


# ── 6. Summary report ────────────────────────────────────────────────────────

def print_summary(mockup_infos: list):
    print("\n" + "="*70)
    print("SUMMARY: Có thể merge warp models an toàn không?")
    print("="*70)
    for view in mockup_infos:
        name = view["name"]
        parts = get_design_parts(view)
        print(f"\n[{name}] — {len(parts)} design parts")
        for p in parts:
            alpha = load_alpha(p["mask_path"])
            binary_ratio = ((alpha < 0.02) | (alpha > 0.98)).mean()
            label = p["name"].split(".")[-1]
            print(f"  {label:30s}  non-zero={( alpha > 0).sum():6d}px  "
                  f"binary={binary_ratio*100:.1f}%  "
                  f"max_alpha={alpha.max():.3f}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    with open(DOWNLOADED_JSON) as f:
        data = json.load(f)
    mockup_infos = data["mockup_infos"]

    print(f"Loaded {len(mockup_infos)} views: {[v['name'] for v in mockup_infos]}")

    print("\n--- [1] Visualise masks per side ---")
    visualise_masks_per_side(mockup_infos)

    print("\n--- [2] Overlap analysis ---")
    analyse_overlap(mockup_infos)

    print("\n--- [3] Seam zone deep-dive ---")
    analyse_seam_zone(mockup_infos)

    print("\n--- [4] Warp coordinate fields ---")
    visualise_warp_fields(mockup_infos)

    print("\n--- [5] Warp discontinuity at seams ---")
    analyse_warp_discontinuity(mockup_infos)

    print("\n--- [6] Summary ---")
    print_summary(mockup_infos)

    print(f"\nAll outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
