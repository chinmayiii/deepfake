# 🧠 Deepfake Detection System

[![CI](https://github.com/chinmayiii/deepfake/actions/workflows/ci.yml/badge.svg)](https://github.com/chinmayiii/deepfake/actions/workflows/ci.yml)

A deepfake detection system built from scratch with PyTorch and EfficientNet-B0, featuring a user-friendly web interface for real-time image and video analysis.

## ⚙️ Built By

- 👨‍💻 [chinmayiii](https://github.com/chinmayiii)

---

## 🌟 Features

- **Deep Learning Model**: EfficientNet-B0 architecture fine-tuned for deepfake detection
- **Multi-format Support**: Analyze both images (.jpg, .jpeg, .png) and videos (.mp4, .mov)
- **Web Interface**: Interactive Gradio-based web application for easy testing
- **Real-time Analysis**: Process first frame of videos for quick deepfake detection
- **Training Pipeline**: Complete PyTorch Lightning training infrastructure
- **Model Export**: Support for PyTorch (.pt) and ONNX format exports
- **Production API**: FastAPI endpoints for health checks, model metadata, single-file inference, and batch inference
- **Inference Metadata**: Per-request UUID, processing latency, model mode/checkpoint, filename hash (SHA-256)
- **Runtime Observability**: In-memory API metrics with per-endpoint request counts, success/failure ratio, and average latency
- **Runtime Introspection**: `/config` endpoint exposes runtime limits and supported formats for client-side validation
- **Admin Metric Controls**: Token-protected `/metrics/reset` endpoint to clear runtime counters safely
- **API Hardening**: Per-IP rate limiting and request timeout guardrails on inference endpoints
- **Probe Coverage**: Dedicated liveness (`/live`) and readiness (`/ready`) probes for deployment orchestration
- **Model Quality Visibility**: `/eval/summary` endpoint serves benchmark/evaluation metrics from a JSON artifact
- **Capability Snapshot**: `/capabilities` endpoint provides one-call summary of probes, inference routes, controls, and evaluation headline
- **Deployment Safety Guards**: Upload validation (type/size), robust error handling, and API-ready JSON responses

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, but recommended for training)

### Installation

1. **Clone the repository:**
   ```bash
  git clone https://github.com/chinmayiii/deepfake.git
   cd DeepfakeDetector
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download a pre-trained model** (or train your own):
   - Place your model file as `models/best_model-v3.pt`

### Usage

#### 🖥️ Web Application

Launch the interactive web interface:

```bash
python web-app.py
```

By default, the server starts on:

```bash
http://127.0.0.1:7860
```

The web app will open in your browser where you can:
- Drag and drop images or videos
- View real-time predictions with confidence scores
- See preview of analyzed content

#### 🔍 Command Line Classification

Classify individual images:

```bash
python classify.py --image path/to/your/image.jpg
```

#### 🎥 Video Analysis

Process videos frame by frame:

```bash
python inference/video_inference.py --video path/to/your/video.mp4
```

#### 🌐 REST API (Deployment)

Once the app is running, these endpoints are available:

- `GET /health` → service + model readiness
- `GET /live` → liveness probe (always `200` when process is up)
- `GET /ready` → readiness probe (`200` when model is ready, `503` otherwise)
- `GET /model-info` → loaded model mode/checkpoint details
- `GET /metrics` → runtime request/latency/success metrics by endpoint
- `GET /config` → runtime limits and accepted extensions
- `GET /capabilities` → one-call service capability and quality summary
- `GET /eval/summary` → latest evaluation metrics artifact (accuracy/F1/ROC-AUC/confusion matrix)
- `POST /metrics/reset` → reset metrics counters (requires `X-Admin-Token` header)
- `POST /detect` → single image/video detection (multipart file)
- `POST /detect/batch` → batch detection for up to 10 files per request

Security note:

- Inference endpoints enforce per-IP request limits and request-level timeout guardrails.

Architecture reference:

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)

Quick checks:

```bash
curl http://127.0.0.1:7860/health
curl http://127.0.0.1:7860/live
curl http://127.0.0.1:7860/ready
curl http://127.0.0.1:7860/model-info
curl http://127.0.0.1:7860/metrics
curl http://127.0.0.1:7860/config
curl http://127.0.0.1:7860/capabilities
curl http://127.0.0.1:7860/eval/summary
```

Environment variables:

- `PORT` (default: `7860`)
- `MAX_UPLOAD_MB` (default: `50`)
- `APP_VERSION` (default: `1.0.0`)
- `METRICS_ADMIN_TOKEN` (optional; required for `/metrics/reset`)
- `RATE_LIMIT_PER_MIN` (default: `60`; per-IP limit for `/detect` and `/detect/batch`)
- `MAX_REQUEST_SECONDS` (default: `30`; timeout guard for inference requests)
- `EVAL_SUMMARY_PATH` (default: `docs/eval_summary.json`; source file for `/eval/summary`)

Generate/update evaluation summary artifact:

```bash
python realeval.py --data path/to/dataset_root --out docs/eval_summary.json
```

Recruiter quick-proof checklist:

- See `docs/RESUME_PROOF.md`

#### 🐳 Docker Deployment

Build and run with Docker:

```bash
docker build -t deepfake-detector .
docker run --rm -p 7860:7860 -e PORT=7860 deepfake-detector
```

#### ☁️ Render Deployment

This repository includes a Render Blueprint config at `render.yaml`.

Deploy steps:

1. Push this project to GitHub.
2. In Render dashboard, click **New +** → **Blueprint**.
3. Select your GitHub repo and deploy.
4. Render will use `render.yaml` automatically.

After deploy, verify:

```bash
curl https://<your-render-domain>/health
curl https://<your-render-domain>/model-info
```

#### ✅ CI Validation

GitHub Actions workflow is included at `.github/workflows/ci.yml` and runs:

- dependency installation from `requirements.txt`
- API smoke tests from `tests/test_api_smoke.py`

Run the same tests locally:

```bash
set SKIP_MODEL_LOAD=1
python -W ignore::ResourceWarning -m unittest discover -s tests -p "test_*.py" -v
```

Windows shortcut:

```bat
run_smoke_tests.bat
```

PowerShell shortcut:

```powershell
./run_smoke_tests.ps1
```

#### 📈 API Benchmark Script

Run a quick latency benchmark on deployed endpoints:

```bash
python tools/benchmark_api.py --base-url http://127.0.0.1:7860 --iterations 20 --output docs/perf_summary.json
```

#### 📊 Validated Runtime Metrics

Latest local benchmark snapshot (`docs/perf_summary.json`, 20 iterations, local run with `SKIP_MODEL_LOAD=1`):

| Endpoint | Status | Avg Latency | P95 Latency |
|---|---:|---:|---:|
| `/live` | 200 | 4.44 ms | 14.11 ms |
| `/ready` | 503 | 2.03 ms | 2.87 ms |
| `/health` | 200 | 2.04 ms | 2.87 ms |
| `/model-info` | 200 | 2.25 ms | 3.84 ms |
| `/config` | 200 | 1.95 ms | 2.45 ms |
| `/capabilities` | 200 | 2.74 ms | 3.92 ms |
| `/eval/summary` | 200 | 9.67 ms | 33.74 ms |
| `/metrics` | 200 | 2.28 ms | 2.65 ms |

Automated API smoke tests: **12/12 passing** (`tests/test_api_smoke.py`).

## 🧾 Resume Highlights (Copy-ready)

- Built and deployed a **hybrid deepfake detection platform** using PyTorch + FastAPI + Gradio with image/video inference support.
- Engineered **production-grade inference APIs** (`/live`, `/ready`, `/health`, `/model-info`, `/detect`, `/detect/batch`) with liveness/readiness probes and batch processing.
- Added **observability metadata** per request (latency, request ID, model checkpoint, SHA-256 file fingerprint) for auditability and reliability.
- Instrumented **service-level API metrics** (`/metrics`) for request volume, success/failure rate, and average latency across endpoints.
- Introduced **structured API error handling** with explicit HTTP status codes and machine-readable error codes for client integration.
- Implemented **safe deployment controls** (strict file validation, request-size limits, structured error handling) to harden real-world usage.
- Containerized deployment with **Docker** and introduced **CI smoke tests** via GitHub Actions for reproducible engineering workflows.
- Added recruiter-facing **capability and proof endpoints** (`/capabilities`, `/eval/summary`) and benchmark artifacts for fast technical validation.

## 🎯 Project Impact (ATS-Optimized)

- Designed and shipped an end-to-end **ML-powered deepfake detection system** with production APIs, real-time inference, and deployment-ready architecture.
- Applied **MLOps fundamentals**: CI validation, containerized packaging, health/readiness endpoints, and environment-driven configuration.
- Improved reliability through **defensive backend engineering** (input validation, bounded batch processing, structured failure paths, and deterministic metadata logging).
- Demonstrated **API observability and performance visibility** with runtime endpoint metrics, benchmarked latency snapshots, and smoke-tested endpoint contracts.
- Added **architecture documentation** with end-to-end request flow and component boundaries to improve engineering communication.
- Built with industry-relevant stack for top SDE/ML roles: **Python, PyTorch, FastAPI, Gradio, Docker, GitHub Actions, OpenCV**.

## 📂 Supported Datasets

This deepfake detection system supports various popular deepfake datasets. Below are the recommended datasets for training and evaluation:

### 🎬 Video-based Datasets

#### **FaceForensics++**
- **Description**: One of the most comprehensive deepfake datasets with 4 manipulation methods
- **Size**: ~1,000 original videos, ~4,000 manipulated videos
- **Manipulations**: Deepfakes, Face2Face, FaceSwap, NeuralTextures
- **Quality**: Raw, c23 (light compression), c40 (heavy compression)
- **Download**: [GitHub Repository](https://github.com/ondyari/FaceForensics)
- **Usage**: Excellent for training robust models across different manipulation types

#### **Celeb-DF (v2)**
- **Description**: High-quality celebrity deepfake dataset
- **Size**: 590 real videos, 5,639 deepfake videos
- **Quality**: High-resolution with improved visual quality
- **Download**: [Official Website](https://github.com/yuezunli/celeb-deepfakeforensics)
- **Usage**: Great for testing model performance on high-quality deepfakes

#### **DFDC (Deepfake Detection Challenge)**
- **Description**: Facebook's large-scale deepfake detection dataset
- **Size**: ~100,000 videos (real and fake)
- **Diversity**: Multiple actors, ethnicities, and ages
- **Download**: [Kaggle Competition](https://www.kaggle.com/c/deepfake-detection-challenge)
- **Usage**: Large-scale training and benchmarking

#### **DFD (Google's Deepfake Detection Dataset)**
- **Description**: Google/Jigsaw deepfake dataset
- **Size**: ~3,000 deepfake videos
- **Quality**: High-quality with various compression levels
- **Download**: [FaceForensics++ repository](https://github.com/ondyari/FaceForensics)
- **Usage**: Additional training data for model robustness

### 🖼️ Image-based Datasets

#### **140k Real and Fake Faces**
- **Description**: Large collection of real and AI-generated face images
- **Size**: ~140,000 images
- **Source**: StyleGAN-generated faces vs real faces
- **Download**: [Kaggle Dataset](https://www.kaggle.com/xhlulu/140k-real-and-fake-faces)
- **Usage**: Perfect for image-based deepfake detection training

#### **CelebA-HQ**
- **Description**: High-quality celebrity face dataset
- **Size**: 30,000 high-resolution images
- **Quality**: 1024×1024 resolution
- **Download**: [GitHub Repository](https://github.com/tkarras/progressive_growing_of_gans)
- **Usage**: Real face examples for training

### 🔧 Dataset Preparation

#### Option 1: Download Pre-processed Datasets
1. Download your chosen dataset from the links above
2. Extract to the `data/` folder
3. Organize as shown in the training section below

#### Option 2: Use Dataset Preparation Tools
Use our built-in tools to prepare datasets:

```bash
# Split video dataset into frames
python tools/split_video_dataset.py --input_dir raw_videos --output_dir data

# Split dataset into train/validation
python tools/split_train_val.py --input_dir data --train_ratio 0.8

# General dataset splitting
python tools/split_dataset.py --input_dir your_dataset --output_dir data
```

### 📋 Dataset Recommendations

- **For Beginners**: Start with **140k Real and Fake Faces** (image-based, easy to work with)
- **For Research**: Use **FaceForensics++** (comprehensive, multiple manipulation types)
- **For Production**: Combine **DFDC** + **Celeb-DF** (large scale, diverse)
- **For High-Quality Testing**: Use **Celeb-DF v2** (challenging, high-quality deepfakes)

### ⚠️ Dataset Usage Notes

- **Ethical Use**: These datasets are for research purposes only
- **Legal Compliance**: Ensure compliance with dataset licenses and terms of use
- **Privacy**: Respect privacy rights of individuals in the datasets
- **Citation**: Properly cite the original dataset papers when publishing research

## 🏋️ Training

### Dataset Structure

Organize your training data in the `data` folder as follows:
```
data/
├── train/
│   ├── real/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── fake/
│       ├── fake1.jpg
│       └── fake2.jpg
└── validation/
    ├── real/
    └── fake/
```

### Configuration

Update `config.yaml` with your dataset paths:

```yaml
train_paths:
  - data/train

val_paths:
  - data/validation

lr: 0.0001
batch_size: 4
num_epochs: 10
```

### Start Training

```bash
python main_trainer.py
```

or

```bash
python model_trainer.py
```

The training will:
- Use PyTorch Lightning for efficient training
- Save best model based on validation loss
- Log metrics to TensorBoard
- Apply early stopping to prevent overfitting

### Monitor Training

View training progress with TensorBoard:

```bash
tensorboard --logdir lightning_logs
```

## 📁 Project Structure

```
├── web-app.py                    # Main web application
├── main_trainer.py               # Primary training script
├── classify.py                   # Image classification utility
├── realeval.py                   # Real-world evaluation script
├── config.yaml                   # Training configuration
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
├── LICENSE                       # MIT License
├── .gitignore                    # Git ignore rules
├── data/                         # Dataset storage (not tracked by git)
│   ├── train/                    # Training data
│   └── validation/               # Validation data
├── datasets/
│   └── hybrid_loader.py          # Custom dataset loader
├── lightning_modules/
│   └── detector.py               # PyTorch Lightning module
├── models/
│   └── best_model-v3.pt          # Trained model weights
├── tools/                        # Dataset preparation utilities
│   ├── split_dataset.py
│   ├── split_train_val.py
│   └── split_video_dataset.py
└── inference/
    ├── export_onnx.py            # ONNX export
    └── video_inference.py        # Video processing
```

## 🛠️ Model Architecture

- **Backbone**: EfficientNet-B0 (pre-trained on ImageNet)
- **Classifier**: Custom 2-class classifier with dropout (0.4)
- **Input Size**: 224x224 RGB images
- **Output**: Binary classification (Real/Fake) with confidence scores

## 📊 Performance

The model achieves:
- High accuracy on diverse deepfake datasets
- Real-time inference capabilities
- Robust performance on compressed/low-quality media

## 🔧 Advanced Usage

### Export to ONNX

Convert PyTorch model to ONNX format:

```bash
python inference/export_onnx.py
```

### Batch Evaluation

Process multiple files programmatically:

```python
from web-app import predict_file

results = []
for file_path in image_paths:
    prediction, confidence, preview = predict_file(file_path)
    results.append({
        'file': file_path,
        'prediction': prediction,
        'confidence': confidence
    })
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- EfficientNet architecture by Google Research
- PyTorch Lightning for training infrastructure
- Gradio for web interface framework
- The research community for deepfake detection advances

---

## 📄 License

This project is licensed under the **MIT License**.

---

⭐ **Star this repository if you found it helpful!**
