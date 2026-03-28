<div align="center">

# 🚗 VLM-LiDAR-Camera-ADAS-Perception

### Zero-Shot Autonomous Driving Scene Understanding with Vision Language Models

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/VasuTammisetti/VLM-LiDAR-Camera-ADAS-perception/blob/main/notebooks/vlm_adas_demo.ipynb)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](docker/)
[![Jenkins](https://img.shields.io/badge/Jenkins-CI%2FCD-D24939?logo=jenkins&logoColor=white)](Jenkinsfile)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**A multi-modal perception system that leverages pre-trained Vision Language Models (VLMs) to analyze driving scenes using Camera and LiDAR data — with zero training, zero annotations, and zero fine-tuning.**

---

</div>

## 🎬 Demo

### RGB Scene Analysis
The model analyzes raw camera images from the KITTI dataset, identifying road users, assessing hazards, and recommending driving actions — all in a single forward pass.

<div align="center">
<img src="outputs/examples/vlm_adas_demo.gif" alt="VLM ADAS Demo" width="900"/>
</div>

### Camera + LiDAR Fusion Analysis
LiDAR point clouds are projected onto the camera image as depth-colored overlays. The VLM uses this fused representation to estimate distances and prioritize hazards by proximity.

<div align="center">
<img src="outputs/examples/vlm_adas_lidar_demo.gif" alt="VLM ADAS LiDAR Demo" width="900"/>
</div>

---

## 💡 Why This Project?

Traditional ADAS perception pipelines require thousands of annotated images, weeks of training, and task-specific model architectures. This project takes a fundamentally different approach:

| Traditional Pipeline | This Project |
|:---:|:---:|
| Thousands of labeled images | **Zero annotations needed** |
| Task-specific model training | **Pre-trained VLM, zero-shot** |
| Separate models per task | **One model, multiple capabilities** |
| Weeks of training | **Inference-only, runs in minutes** |
| Fixed output categories | **Free-form natural language analysis** |

---

## 🏗️ Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                    Input Pipeline                               │
│                                                                 │
│   KITTI Camera ──────┐                                          │
│   (RGB Image)        ├──► Image Preprocessing ──┐               │
│                      │                          │               │
│   KITTI LiDAR ───────┘                          │               │
│   (Velodyne .bin)                               │               │
│       │                                         ▼               │
│       ├──► Depth Projection ──► RGB+Depth ──► VLM Engine        │
│       │    (Calib P2,R0,Tr)     Overlay       (LLaVA-1.6        │
│       │                                       Mistral-7B        │
│       └──► BEV Generation ──► Bird's Eye       4-bit NF4)       │
│                                View              │              │
│                                                  ▼              │
│                                         Structured Output       │
│                                          ├─ Scene Context       │
│                                          ├─ Object Detection    │
│                                          ├─ Hazard Assessment   │
│                                          └─ Drive Recommendation│
└─────────────────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

- **Zero-Shot Scene Analysis** — No training, no annotations. The pre-trained VLM understands driving scenes out of the box using carefully engineered ADAS-specific prompts.

- **Camera-LiDAR Fusion** — Velodyne 3D point clouds are projected onto the 2D camera image using KITTI calibration matrices (P2, R0_rect, Tr_velo_to_cam), creating depth-aware visual inputs.

- **Bird's Eye View** — LiDAR data is transformed into a top-down BEV representation for spatial awareness of the driving environment.

- **Multi-Prompt Pipeline** — Four specialized prompt modes: full scene analysis, hazard-only detection, depth-aware analysis, and object counting.

- **4-bit Quantization** — Runs on consumer GPUs (RTX 2070, 8GB VRAM) using NF4 quantization via bitsandbytes, making it accessible without cloud infrastructure.

- **CI/CD Pipeline** — Dockerized testing and deployment with Jenkins, including GPU and CPU-only configurations.

---

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)
Click the badge above or run:
```bash
# Opens directly in Colab with free T4 GPU
```
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/VasuTammisetti/VLM-LiDAR-Camera-ADAS-perception/blob/main/notebooks/vlm_adas_demo.ipynb)

### Option 2: Local Setup (RTX 2070+ / 8GB VRAM)
```bash
git clone https://github.com/VasuTammisetti/VLM-LiDAR-Camera-ADAS-perception.git
cd VLM-LiDAR-Camera-ADAS-perception

python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\\Scripts\\activate         # Windows

pip install -r requirements.txt
python run_demo.py --env local --model llava-1.5-7b --num_scenes 5
```

### Option 3: Docker
```bash
# Run inference (requires NVIDIA Docker runtime)
docker compose run inference

# Run tests (no GPU needed)
docker compose run test

# Run linting
docker compose run lint
```

---

## 📁 Project Structure
```
VLM-LiDAR-Camera-ADAS-perception/
├── src/
│   ├── config.py              # Environment-aware data paths
│   ├── model_loader.py        # VLM loading with 4-bit quantization
│   ├── scene_analyzer.py      # ADAS prompt templates + inference
│   └── visualization.py       # LiDAR projection, BEV, result display
├── tests/
│   ├── test_model_loader.py   # Model loading validation
│   ├── test_scene_analyzer.py # Prompt integrity tests
│   └── test_visualization.py  # LiDAR processing tests
├── docker/
│   ├── Dockerfile             # GPU inference container
│   └── Dockerfile.test        # Lightweight CI container
├── outputs/examples/
│   ├── vlm_adas_demo.gif      # RGB analysis demo
│   └── vlm_adas_lidar_demo.gif # LiDAR fusion demo
├── data/sample_scenes/        # Sample KITTI frames for testing
├── Jenkinsfile                # CI/CD pipeline definition
├── docker-compose.yml         # Multi-service orchestration
├── run_demo.py                # Main entry point
├── generate_demo_gif.py       # Demo GIF generator
├── requirements.txt           # Production dependencies
└── requirements-dev.txt       # Development dependencies
```

---

## 🔧 ADAS Prompt Engineering

The core innovation is in the prompt design — transforming a general-purpose VLM into a domain-specific ADAS perception system:

| Prompt Mode | Purpose | Use Case |
|:---|:---|:---|
| `full_analysis` | Complete scene breakdown: context, objects, hazards, recommendations | General driving analysis |
| `hazard_only` | Risk-focused detection with severity ranking | Safety-critical assessment |
| `depth_aware` | Distance estimation using LiDAR depth overlay colors | Proximity-based hazard priority |
| `object_count` | Exhaustive object enumeration with positions | Perception validation |

---

## 🧠 Models

| Model | VRAM | Quantization | Best For |
|:---|:---:|:---:|:---|
| **LLaVA-1.6-Mistral-7B** | ~5-6 GB | 4-bit NF4 | Detailed structured analysis |
| PaliGemma-3B | ~3-4 GB | 4-bit NF4 | Short captioning tasks |

---

## 🔄 CI/CD Pipeline
```
git push ──► Jenkins ──► Build Test Container (no GPU)
                              │
                              ├──► Lint (flake8)
                              ├──► Unit Tests (pytest, 11 tests)
                              │
                              └──► Build App Image ──► Push to DockerHub
```

All tests run **without a GPU** — they validate data loading, calibration parsing, prompt structure, and projection math. VLM inference is a separate GPU-dependent step.

---

## 📊 Technical Details

- **LiDAR Projection**: 3D Velodyne points → 2D image plane using KITTI calibration matrices (P2 × R0_rect × Tr_velo_to_cam)
- **Depth Visualization**: Points colored by distance — blue (0-10m), green (10-25m), red (25-50m)
- **BEV Generation**: Top-down view with configurable range (default: 40m × 40m, 0.1m resolution)
- **Memory Optimization**: 4-bit NF4 quantization reduces 7B model from ~14GB to ~5GB VRAM

---

## 📚 Dataset

This project uses the [KITTI Vision Benchmark Suite](http://www.cvlibs.net/datasets/kitti/):
- **Left Camera Images** (RGB, 1242×375)
- **Velodyne LiDAR** (64-beam, ~120K points/frame)
- **Calibration Files** (camera intrinsics/extrinsics, LiDAR-camera transform)

No annotations are used — this is a **zero-shot inference** project.

---

## 🛠️ Tech Stack

`Python` `PyTorch` `HuggingFace Transformers` `LLaVA` `bitsandbytes` `KITTI` `Docker` `Jenkins` `NumPy` `Matplotlib`

---

## 👤 Author

**Vasu Tammisetti**
AI Research Engineer & Doctoral Researcher at Infineon Technologies AG, Munich.
PhD: Meta-Learning for ADAS Perception — University of Granada.

[![GitHub](https://img.shields.io/badge/GitHub-VasuTammisetti-181717?logo=github)](https://github.com/VasuTammisetti)

---

## 📝 License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">

**If you find this project useful, please ⭐ star the repository!**

</div>
