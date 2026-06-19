#!/usr/bin/env python3
"""
TDD: Seam quality test for optimized mockup generation.

Checks that the optimized output has no jagged edges at seam regions
(where multiple warp parts' masks overlap).

Usage:
  python test_seam_quality.py [--generate]
    --generate  re-run mockup generation before testing
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
import do_optimize as mo

ROOT = mo.ROOT_DIR
PREFIX = mo.PREFIX
OUTPUT_BASE = f"{ROOT}/{PREFIX}/mockups-output"

VIEWS = ["Front", "Back", "Front_Back"]
ARTWORKS = [
    "5c25c2cb-decb-4b6e-9865-28aa8eda2ea9",
    "3f99eec66e7d5230ac77",
]

# ── helpers ──────────────────────────────────────────────────────────────────

def load_rgb(path):
    return np.array(Image.open(path).convert('RGBA'), dtype=np.uint8)


def pixel_diff(a, b):
    """Per-pixel max-channel absolute diff (uint8 → int)."""
    return np.abs(a[:, :, :3].astype(np.int16) - b[:, :, :3].astype(np.int16)).max(axis=2).astype(np.uint16)


def seam_mask_from_parts(sub_parts):
    """Pixels where ≥2 sub-parts have alpha > 128 (potential seam region)."""
    overlap = np.zeros((1000, 1000), dtype=np.int16)
    for p in sub_parts:
        a = np.array(Image.open(p["mask_path"]).convert("RGBA"))[:, :, 3]
        overlap += (a > 128).astype(np.int16)
    return overlap >= 2


def jagged_ratio_at_seam(opt_img, seam):
    """
    Staircase detection at seam pixels.
    A jagged boundary has large one-axis gradient but no cross-axis gradient at the SAME pixel.
    Returns (jagged_ratio, edge_pixel_count).
    """
    gray = opt_img[:, :, :3].mean(axis=2).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)

    edge = (mag > 15) & seam
    n_edge = edge.sum()
    if n_edge == 0:
        return 0.0, 0

    # Pure-axis edge pixels: dominated by x OR y, not diagonal
    gx_dom = (np.abs(gx) > 4 * np.abs(gy)) & edge
    gy_dom = (np.abs(gy) > 4 * np.abs(gx)) & edge

    # Staircase: neighbour has opposite-axis dominance
    # Dilate each mask by 1px and check intersection
    kern = np.ones((3, 3), np.uint8)
    gx_dilated = cv2.dilate(gx_dom.astype(np.uint8), kern) > 0
    gy_dilated = cv2.dilate(gy_dom.astype(np.uint8), kern) > 0

    staircase = ((gx_dom & gy_dilated) | (gy_dom & gx_dilated)) & edge
    return staircase.sum() / n_edge, n_edge


# ── test cases ───────────────────────────────────────────────────────────────

class TestResult:
    def __init__(self, name, passed, msg):
        self.name = name
        self.passed = passed
        self.msg = msg

    def __repr__(self):
        s = "PASS" if self.passed else "FAIL"
        return f"[{s}] {self.name}: {self.msg}"


THRESHOLDS = {
    # npy 1000×1000: full-resolution grids, output should be near-identical to original
    "optimized": {
        "overall_diff": 0.03,   # <3% of garment pixels differ >20
        "seam_diff":    0.05,   # <5% of seam pixels differ >20
        "staircase":    0.20,   # <20% staircase ratio at seam edges
    },
    # json 100×100: downsampled grids — inherently approximate.
    # 100→1000 bilinear upsample introduces ~5-10px coord error; for artworks
    # with sharp color transitions this causes smooth tone shifts (not jagged).
    # Thresholds are looser; staircase test is the primary quality gate.
    "optimized_100x100": {
        "overall_diff": 0.15,   # allow up to 15% (smooth approximation, not artifacts)
        "seam_diff":    0.30,   # allow up to 30% (inherent 100×100 approximation)
        "staircase":    0.20,   # same jagged-edge guard as npy path
    },
}


def run_all_tests(config_type="optimized"):
    opt_path = f"{ROOT}/{PREFIX}/mockup_infos.optimized.json"
    with open(opt_path) as f:
        opt_infos = json.load(f)

    th = THRESHOLDS.get(config_type, THRESHOLDS["optimized"])
    results = []

    for artwork in ARTWORKS:
        for view_name in VIEWS:
            orig_path = f"{OUTPUT_BASE}/original/{artwork}_{view_name}.png"
            opt_path_img = f"{OUTPUT_BASE}/{config_type}/{artwork}_{view_name}.png"

            if not os.path.exists(orig_path) or not os.path.exists(opt_path_img):
                results.append(TestResult(f"[{view_name}/{artwork[:8]}] files", False, "images not found"))
                continue

            orig = load_rgb(orig_path)
            opt  = load_rgb(opt_path_img)

            view_info = next((v for v in opt_infos["mockup_infos"] if v["name"] == view_name), None)
            if view_info is None:
                continue

            design_parts = [p for p in view_info["parts"] if p.get("warp_type") == "warp_npy"]
            sub_parts = []
            for dp in design_parts:
                wi = dp.get("warp_info", {})
                sub_parts.extend(wi.get("parts", wi.get("parts_json", [])))

            garment  = orig[:, :, 3] > 10
            total_g  = garment.sum()
            if total_g == 0:
                continue

            diff = pixel_diff(orig, opt)

            # ── T1: overall diff ──────────────────────────────────────────
            diff_ratio = (diff[garment] > 20).sum() / total_g
            results.append(TestResult(
                f"[{view_name}/{artwork[:8]}] overall_diff",
                diff_ratio < th["overall_diff"],
                f"{diff_ratio:.2%} of garment pixels differ >20  (limit {th['overall_diff']:.0%})",
            ))

            if sub_parts:
                seam    = seam_mask_from_parts(sub_parts)
                seam_px = seam.sum()

                # ── T2: seam diff ─────────────────────────────────────────
                if seam_px > 0:
                    seam_diff_ratio = (diff[seam] > 20).sum() / seam_px
                    results.append(TestResult(
                        f"[{view_name}/{artwork[:8]}] seam_diff",
                        seam_diff_ratio < th["seam_diff"],
                        f"{seam_diff_ratio:.2%} of {seam_px} seam px differ >20  (limit {th['seam_diff']:.0%})",
                    ))

                # ── T3: staircase (primary quality gate) ──────────────────
                ratio, n_edge = jagged_ratio_at_seam(opt, seam)
                results.append(TestResult(
                    f"[{view_name}/{artwork[:8]}] staircase",
                    ratio < th["staircase"],
                    f"staircase={ratio:.2%} over {n_edge} seam-edge px  (limit {th['staircase']:.0%})",
                ))

    return results


# ── main ─────────────────────────────────────────────────────────────────────

def print_results(title, results):
    pad = max(len(r.name) for r in results) + 2
    print(f"\n{'='*70}")
    print(f" {title}")
    print(f"{'='*70}")
    for r in results:
        print(f"  {repr(r)}")
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    print(f"{'='*70}")
    print(f"  {passed}/{total} passed")
    return passed == total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true", help="Re-run mockup generation first")
    args = parser.parse_args()

    if args.generate:
        print("Re-generating mockups…")
        mo.run_mockup_generation()

    ok_npy = print_results("optimized (npy 1000×1000)", run_all_tests("optimized"))
    ok_json = print_results("optimized_100x100 (json)", run_all_tests("optimized_100x100"))

    sys.exit(0 if (ok_npy and ok_json) else 1)


if __name__ == "__main__":
    main()
