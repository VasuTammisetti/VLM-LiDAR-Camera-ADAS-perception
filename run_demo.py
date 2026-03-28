"""
Main demo script — run VLM-based ADAS scene analysis.
Usage:
    python run_demo.py                          # uses first 5 images
    python run_demo.py --model llava-1.5-7b     # use LLaVA instead
    python run_demo.py --env local              # local data paths
    python run_demo.py --num_scenes 10          # analyze 10 scenes
"""
import argparse
import glob
import os
import torch
from PIL import Image

from src.config import get_paths
from src.model_loader import load_model
from src.scene_analyzer import analyze_scene
from src.visualization import (
    create_lidar_overlay,
    create_combined_view,
    display_results,
)


def main(args):
    # ── Setup ──
    paths = get_paths(args.env)
    os.makedirs("outputs/examples", exist_ok=True)

    # ── Load model ──
    print(f"\n{'='*50}")
    print(f"  VLM-ADAS Scene Understanding Demo")
    print(f"  Model: {args.model}")
    print(f"{'='*50}\n")

    model, processor = load_model(args.model)

    # ── Get image list ──
    image_files = sorted(glob.glob(os.path.join(paths["images"], "*.png")))[:args.num_scenes]
    print(f"\n📸 Found {len(image_files)} scenes to analyze\n")

    # ── Process each scene ──
    for i, img_path in enumerate(image_files):
        frame_id = os.path.splitext(os.path.basename(img_path))[0]
        print(f"\n{'─'*40}")
        print(f"  Scene {i+1}/{len(image_files)}: {frame_id}")
        print(f"{'─'*40}")

        image = Image.open(img_path).convert("RGB")

        # Step 1: VLM analysis on raw RGB
        print("  🔍 Running VLM analysis...")
        result = analyze_scene(image, model, processor, prompt_type="full_analysis")
        print(f"\n{result}\n")

        # Step 2: Save result visualization
        save_path = f"outputs/examples/{frame_id}_analysis.png"
        display_results(image, result, title=f"Scene {frame_id}", save_path=save_path)

        # Step 3: LiDAR overlay (if velodyne data exists)
        vel_path = os.path.join(paths["velodyne"], f"{frame_id}.bin")
        cal_path = os.path.join(paths["calib"], f"{frame_id}.txt")

        if os.path.exists(vel_path) and os.path.exists(cal_path):
            print("  📡 Creating LiDAR overlay...")
            overlay_path = f"outputs/examples/{frame_id}_lidar_overlay.png"
            create_lidar_overlay(img_path, vel_path, cal_path, save_path=overlay_path)

            # Step 4: Depth-aware VLM analysis
            overlay_img = Image.open(overlay_path).convert("RGB")
            print("  🔍 Running depth-aware VLM analysis...")
            depth_result = analyze_scene(overlay_img, model, processor, prompt_type="depth_aware")
            print(f"\n{depth_result}\n")

            # Step 5: Combined 3-panel view
            combined_path = f"outputs/examples/{frame_id}_combined.png"
            create_combined_view(img_path, vel_path, cal_path, save_path=combined_path)
        else:
            print("  ⚠️  No LiDAR/calib data found, skipping depth analysis")

        # Free VRAM
        torch.cuda.empty_cache()

    # ── Summary ──
    print(f"\n{'='*50}")
    print(f"  ✅ Done! Results saved to outputs/examples/")
    print(f"  Peak VRAM: {torch.cuda.max_memory_allocated() / 1024**3:.1f} GB")
    print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLM-ADAS Scene Understanding")
    parser.add_argument("--model", default="paligemma-3b", choices=["paligemma-3b", "llava-1.5-7b"])
    parser.add_argument("--env", default=None, help="colab, local, or docker")
    parser.add_argument("--num_scenes", type=int, default=5)
    args = parser.parse_args()
    main(args)