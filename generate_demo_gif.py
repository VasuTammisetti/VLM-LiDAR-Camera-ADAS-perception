"""
Generate demo GIF: VLM analysis on KITTI test scenes.
"""
import torch
import glob
import os
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model_loader import load_model
from src.scene_analyzer import analyze_scene
from src.visualization import load_velodyne, load_calib, project_lidar_to_image
import numpy as np


def create_analysis_frame(image, analysis_text, frame_id, frame_num, total):
    """Create a single frame: RGB + VLM Analysis side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7),
                                    gridspec_kw={"width_ratios": [1, 1]})
    fig.patch.set_facecolor('#1a1a2e')

    # Left: Input image
    ax1.imshow(image)
    ax1.set_title(f"Input Scene: {frame_id}", fontsize=14, color='white', fontweight='bold')
    ax1.axis('off')

    # Right: VLM Analysis
    ax2.set_facecolor('#16213e')
    ax2.text(0.05, 0.95, analysis_text, transform=ax2.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             color='#e0e0e0', wrap=True,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#0f3460', alpha=0.9))
    ax2.set_title("VLM Scene Analysis", fontsize=14, color='white', fontweight='bold')
    ax2.axis('off')

    # Header
    fig.suptitle(f"VLM-ADAS Scene Understanding  [{frame_num}/{total}]",
                 fontsize=18, color='#e94560', fontweight='bold', y=0.98)

    # Footer
    fig.text(0.5, 0.01, "Model: LLaVA-1.6-Mistral-7B (4-bit) | Data: KITTI | Zero-Shot Inference",
             ha='center', fontsize=10, color='#888888')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save to buffer
    save_path = f"outputs/examples/frame_{frame_id}.png"
    fig.savefig(save_path, dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    return save_path


def create_lidar_frame(image_path, vel_path, cal_path, analysis_text, frame_id, frame_num, total):
    """Create frame with LiDAR overlay + analysis."""
    img = np.array(Image.open(image_path))
    h, w = img.shape[:2]

    points = load_velodyne(vel_path)
    P2, R0, Tr = load_calib(cal_path)
    u, v, depth = project_lidar_to_image(points, P2, R0, Tr, w, h)

    fig, axes = plt.subplots(1, 3, figsize=(24, 6),
                              gridspec_kw={"width_ratios": [1, 1, 1]})
    fig.patch.set_facecolor('#1a1a2e')

    # Panel 1: RGB
    axes[0].imshow(img)
    axes[0].set_title("Front Camera", fontsize=13, color='white', fontweight='bold')
    axes[0].axis('off')

    # Panel 2: LiDAR overlay
    axes[1].imshow(img)
    axes[1].scatter(u, v, c=depth, cmap='jet', s=1, alpha=0.7, vmin=0, vmax=50)
    axes[1].set_title("RGB + LiDAR Depth", fontsize=13, color='white', fontweight='bold')
    axes[1].axis('off')

    # Panel 3: Analysis
    axes[2].set_facecolor('#16213e')
    axes[2].text(0.05, 0.95, analysis_text, transform=axes[2].transAxes,
                 fontsize=9, verticalalignment='top', fontfamily='monospace',
                 color='#e0e0e0', wrap=True,
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#0f3460', alpha=0.9))
    axes[2].set_title("VLM Analysis", fontsize=13, color='white', fontweight='bold')
    axes[2].axis('off')

    fig.suptitle(f"VLM-ADAS Scene Understanding  [{frame_num}/{total}]",
                 fontsize=16, color='#e94560', fontweight='bold', y=0.98)
    fig.text(0.5, 0.01, "Model: LLaVA-1.6-Mistral-7B (4-bit) | Data: KITTI | Camera + LiDAR Fusion",
             ha='center', fontsize=10, color='#888888')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    save_path = f"outputs/examples/lidar_frame_{frame_id}.png"
    fig.savefig(save_path, dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    return save_path


def main():
    os.makedirs("outputs/examples", exist_ok=True)

    # ─── Load model ───
    print("=" * 50)
    print("  VLM-ADAS Demo GIF Generator")
    print("=" * 50)
    model, processor = load_model("llava-1.5-7b")

    # ─── Get test images (20 scenes) ───
    test_img_dir = "/content/sensorfusion/sensorfusion/data_object_image_2/testing"
    test_vel_dir = "/content/sensorfusion/sensorfusion/data_object_velodyne/testing"
    test_cal_dir = "/content/sensorfusion/sensorfusion/data_object_calib/testing"

    test_images = sorted(glob.glob(os.path.join(test_img_dir, "*.png")))[:20]
    total = len(test_images)
    print(f"\nProcessing {total} test scenes...\n")

    rgb_frames = []
    lidar_frames = []

    for i, img_path in enumerate(test_images):
        frame_id = os.path.splitext(os.path.basename(img_path))[0]
        print(f"[{i+1}/{total}] Analyzing {frame_id}...")

        image = Image.open(img_path).convert("RGB")

        # VLM analysis
        result = analyze_scene(image, model, processor, prompt_type="full_analysis")
        # Truncate for display
        if len(result) > 600:
            result = result[:600] + "..."

        print(f"  Analysis: {result[:80]}...")

        # RGB + Analysis frame
        rgb_path = create_analysis_frame(image, result, frame_id, i+1, total)
        rgb_frames.append(rgb_path)

        # LiDAR frame (if data exists)
        vel_path = os.path.join(test_vel_dir, f"{frame_id}.bin")
        cal_path = os.path.join(test_cal_dir, f"{frame_id}.txt")

        if os.path.exists(vel_path) and os.path.exists(cal_path):
            lidar_path = create_lidar_frame(
                img_path, vel_path, cal_path, result, frame_id, i+1, total
            )
            lidar_frames.append(lidar_path)

        torch.cuda.empty_cache()

    # ─── Create GIFs ───
    print("\nCreating GIFs...")

    # RGB Analysis GIF
    if rgb_frames:
        frames = [Image.open(f) for f in rgb_frames]
        frames[0].save(
            "outputs/examples/vlm_adas_demo.gif",
            save_all=True,
            append_images=frames[1:],
            duration=2000,  # 2 seconds per frame
            loop=0
        )
        size = os.path.getsize("outputs/examples/vlm_adas_demo.gif") / 1024**2
        print(f"  vlm_adas_demo.gif ({size:.1f} MB, {len(frames)} frames, 2s each)")

    # LiDAR Analysis GIF
    if lidar_frames:
        frames = [Image.open(f) for f in lidar_frames]
        frames[0].save(
            "outputs/examples/vlm_adas_lidar_demo.gif",
            save_all=True,
            append_images=frames[1:],
            duration=2000,
            loop=0
        )
        size = os.path.getsize("outputs/examples/vlm_adas_lidar_demo.gif") / 1024**2
        print(f"  vlm_adas_lidar_demo.gif ({size:.1f} MB, {len(frames)} frames, 2s each)")

    print(f"\nPeak VRAM: {torch.cuda.max_memory_allocated() / 1024**3:.1f} GB")
    print("Done!")


if __name__ == "__main__":
    main()
