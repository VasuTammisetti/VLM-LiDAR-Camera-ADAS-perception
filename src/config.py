"""
Data paths and configuration for different environments.

Reproducibility note: every environment falls back to the bundled
`data/sample_scenes` directory when the full KITTI dataset is not present,
so a fresh clone (or a Colab opened by anyone) runs out of the box.
"""
import os

COLAB_BASE = "/content/sensorfusion/sensorfusion"

DATA_PATHS = {
    "colab": {
        "images": os.path.join(COLAB_BASE, "data_object_image_2/training"),
        "velodyne": os.path.join(COLAB_BASE, "data_object_velodyne/training"),
        "calib": os.path.join(COLAB_BASE, "data_object_calib/training"),
    },
    "local": {
        "images": "data/image_2",
        "velodyne": "data/velodyne",
        "calib": "data/calib",
    },
    "docker": {
        "images": "/app/data/image_2",
        "velodyne": "/app/data/velodyne",
        "calib": "/app/data/calib",
    },
}

# Bundled sample data — always present in the repo, used as a fallback
SAMPLE_DIR = "data/sample_scenes"

# Model configs optimized for different GPUs
MODEL_CONFIGS = {
    "llava-1.5-7b": {
        "model_id": "llava-hf/llava-v1.6-mistral-7b-hf",
        "min_vram_gb": 6,
        "description": "Default. Open access, handles structured ADAS prompts.",
    },
    "paligemma-3b": {
        "model_id": "google/paligemma-3b-mix-448",
        "min_vram_gb": 4,
        "description": "Smaller, but GATED — requires HuggingFace login + license.",
    },
}

OUTPUT_DIR = "outputs/examples"


def get_paths(env=None):
    """
    Resolve data paths for the given environment.

    Auto-detects the environment when env is None. If the full dataset for
    the detected environment is missing, falls back to the bundled
    sample_scenes so the demo always has images to run on.
    """
    if env is None:
        if os.path.exists(COLAB_BASE):
            env = "colab"
        elif os.path.exists("/app/data"):
            env = "docker"
        else:
            env = "local"

    paths = dict(DATA_PATHS[env])  # copy so we can safely override

    # Fallback: if the full image set is missing, use bundled samples
    if not (os.path.exists(paths["images"]) and os.listdir(paths["images"])):
        if os.path.exists(SAMPLE_DIR) and os.listdir(SAMPLE_DIR):
            print(f"Full dataset not found for env='{env}'. "
                  f"Falling back to bundled samples: {SAMPLE_DIR}")
            paths = {
                "images": SAMPLE_DIR,
                "velodyne": "data/velodyne",   # only used if present
                "calib": "data/calib",         # only used if present
            }

    print(f"Environment: {env}")
    for name, path in paths.items():
        exists = os.path.exists(path)
        count = len(os.listdir(path)) if exists else 0
        status = f"{count} files" if exists else "not found"
        print(f"   {name}: {status}")
    return paths
