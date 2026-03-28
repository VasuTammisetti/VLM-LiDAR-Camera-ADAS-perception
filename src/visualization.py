"""
LiDAR projection, BEV generation, and result visualization.
"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os


# ─── LiDAR Utilities ─────────────────────────────────────

def load_velodyne(bin_path):
    """Load Velodyne point cloud from .bin file. Returns Nx3 (x,y,z)."""
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return points[:, :3]


def load_calib(calib_path):
    """Parse KITTI calibration file. Returns P2, R0_rect, Tr_velo_to_cam."""
    calib = {}
    with open(calib_path, 'r') as f:
        for line in f:
            if ':' not in line:
                continue
            key, *vals = line.strip().split()
            key = key.rstrip(':')
            calib[key] = np.array([float(v) for v in vals])

    P2 = calib['P2'].reshape(3, 4)
    R0 = np.eye(4)
    R0[:3, :3] = calib['R0_rect'].reshape(3, 3)
    Tr = np.eye(4)
    Tr[:3, :4] = calib['Tr_velo_to_cam'].reshape(3, 4)
    return P2, R0, Tr


def project_lidar_to_image(points_3d, P2, R0, Tr, img_w, img_h):
    """Project 3D LiDAR points onto 2D image plane."""
    n = points_3d.shape[0]
    pts_hom = np.hstack([points_3d, np.ones((n, 1))])
    pts_cam = (P2 @ R0 @ Tr @ pts_hom.T).T

    depth = pts_cam[:, 2]
    mask = depth > 0

    u = (pts_cam[:, 0] / depth).astype(int)
    v = (pts_cam[:, 1] / depth).astype(int)
    mask &= (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)

    return u[mask], v[mask], depth[mask]


# ─── Visualization Functions ─────────────────────────────

def create_lidar_overlay(image_path, velodyne_path, calib_path, save_path=None):
    """
    Project LiDAR points onto camera image, colored by depth.
    Returns PIL Image of the overlay.
    """
    img = np.array(Image.open(image_path))
    h, w = img.shape[:2]

    points = load_velodyne(velodyne_path)
    P2, R0, Tr = load_calib(calib_path)
    u, v, depth = project_lidar_to_image(points, P2, R0, Tr, w, h)

    fig, ax = plt.subplots(1, 1, figsize=(14, 5))
    ax.imshow(img)
    scatter = ax.scatter(u, v, c=depth, cmap='jet', s=1, alpha=0.7, vmin=0, vmax=50)
    plt.colorbar(scatter, ax=ax, label='Depth (m)', shrink=0.7)
    ax.set_title('RGB + LiDAR Depth Overlay')
    ax.axis('off')

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()
    plt.close()

    if save_path:
        return Image.open(save_path)
    return None


def create_bev(velodyne_path, x_range=(-20, 20), y_range=(0, 40), res=0.1):
    """
    Create Bird's Eye View image from LiDAR point cloud.
    Returns 2D numpy array.
    """
    points = load_velodyne(velodyne_path)

    mask = (
        (points[:, 0] > x_range[0]) & (points[:, 0] < x_range[1]) &
        (points[:, 1] > y_range[0]) & (points[:, 1] < y_range[1])
    )
    pts = points[mask]

    bev_h = int((y_range[1] - y_range[0]) / res)
    bev_w = int((x_range[1] - x_range[0]) / res)
    bev = np.zeros((bev_h, bev_w), dtype=np.float32)

    xi = ((pts[:, 0] - x_range[0]) / res).astype(int)
    yi = ((pts[:, 1] - y_range[0]) / res).astype(int)
    yi = bev_h - 1 - yi

    valid = (xi >= 0) & (xi < bev_w) & (yi >= 0) & (yi < bev_h)
    bev[yi[valid], xi[valid]] = 1.0

    return bev


def create_combined_view(image_path, velodyne_path, calib_path, save_path=None):
    """Create side-by-side: RGB + LiDAR overlay + BEV."""
    img = Image.open(image_path)

    points = load_velodyne(velodyne_path)
    P2, R0, Tr = load_calib(calib_path)
    img_arr = np.array(img)
    h, w = img_arr.shape[:2]
    u, v, depth = project_lidar_to_image(points, P2, R0, Tr, w, h)

    bev = create_bev(velodyne_path)

    fig, axes = plt.subplots(1, 3, figsize=(22, 5))

    # Panel 1: RGB
    axes[0].imshow(img)
    axes[0].set_title('Front Camera (RGB)')
    axes[0].axis('off')

    # Panel 2: LiDAR overlay
    axes[1].imshow(img_arr)
    axes[1].scatter(u, v, c=depth, cmap='jet', s=1, alpha=0.7, vmin=0, vmax=50)
    axes[1].set_title('RGB + LiDAR Depth')
    axes[1].axis('off')

    # Panel 3: BEV
    axes[2].imshow(bev, cmap='hot', origin='lower')
    axes[2].set_title("Bird's Eye View (LiDAR)")
    axes[2].set_xlabel('Lateral (m)')
    axes[2].set_ylabel('Longitudinal (m)')

    plt.suptitle('Multi-Modal ADAS Perception', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()
    plt.close()

    if save_path:
        return Image.open(save_path)
    return None


def display_results(image, analysis_text, title="VLM Scene Analysis", save_path=None):
    """Display input image alongside VLM analysis text."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6),
                                    gridspec_kw={'width_ratios': [1, 1]})

    ax1.imshow(image)
    ax1.set_title("Input Scene", fontsize=13)
    ax1.axis('off')

    ax2.text(0.05, 0.95, analysis_text, transform=ax2.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             wrap=True, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax2.set_title("VLM Analysis", fontsize=13)
    ax2.axis('off')

    plt.suptitle(title, fontsize=15, fontweight='bold')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()
    plt.close()