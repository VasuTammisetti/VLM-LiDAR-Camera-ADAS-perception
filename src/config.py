"""
Data paths and configuration for different environments.
"""
import os

COLAB_BASE = "/content/sensorfusion/sensorfusion"

DATA_PATHS = {
    "colab": {
        "images":   os.path.join(COLAB_BASE, "data_object_image_2/training"),
        "velodyne": os.path.join(COLAB_BASE, "data_object_velodyne/training"),
        "calib":    os.path.join(COLAB_BASE, "data_object_calib/training"),
    },
    "local": {
        "images":   "data/image_2",
        "velodyne": "data/velodyne",
        "calib":    "data/calib",
    },
    "docker": {
        "images":   "/app/data/image_2",
        "velodyne": "/app/data/velodyne",
        "calib":    "/app/data/calib",
    }
}

# Model configs optimized for different GPUs
MODEL_CONFIGS = {
    "paligemma-3b": {
        "model_id": "google/paligemma-3b-mix-448",
        "min_vram_gb": 4,
        "description": "Best for 8GB GPUs (RTX 2070)"
    },
    "llava-1.5-7b": {
        "model_id": "llava-hf/llava-v1.6-mistral-7b-hf",
        "min_vram_gb": 6,
        "description": "Better quality, needs 8GB+ VRAM"
    }
}

OUTPUT_DIR = "outputs/examples"
SAMPLE_DIR = "data/sample_scenes"

def get_paths(env=None):
    """Auto-detect environment or use specified one."""
    if env is None:
        if os.path.exists("/content"):
            env = "colab"
        elif os.path.exists("/app/data"):
            env = "docker"
        else:
            env = "local"

    paths = DATA_PATHS[env]
    print(f"📍 Environment: {env}")
    for name, path in paths.items():
        exists = os.path.exists(path)
        count = len(os.listdir(path)) if exists else 0
        status = f"✅ {count} files" if exists else "❌ not found"
        print(f"   {name}: {status}")
    return paths