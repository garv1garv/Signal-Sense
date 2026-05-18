# 👁️ SignalSense AI

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.112-009688.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3.0-EE4C2C.svg)
![Celery](https://img.shields.io/badge/Celery-5.4.0-37814A.svg)
![Optuna](https://img.shields.io/badge/Optuna-3.6.1-blue.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)

**SignalSense AI** is a production-grade, distributed AI video understanding and anomaly detection platform. It leverages a powerful ensemble of Vision and Large Language Models (LLMs) to perform complex temporal event detection, zero-shot scene classification, and natural language video narration. 

Designed for high-throughput and continuous self-improvement, the platform features a zero-downtime hot-swapping architecture and fully automated nightly hyperparameter optimization (HPO).

---

## ✨ Key Features

- **🧠 Multi-Model AI Pipeline**:
  - **YOLOv9**: High-speed frame-by-frame object detection.
  - **OpenAI CLIP**: Zero-shot scene classification and temporal feature extraction.
  - **DINOv2**: State-of-the-art spatial embeddings for tracking and structural understanding.
  - **Phi-3.5-Vision**: Local, privacy-preserving multimodal LLM used to generate synthetic conversational data and human-readable video narratives.
- **⚡ Distributed Architecture**: Asynchronous video processing powered by **Celery** and **Redis**. Built to scale horizontally across multiple GPU worker nodes.
- **🔄 Zero-Downtime Hot-Swapping**: Update model configurations, thresholds, and LoRA weights on the fly without dropping a single active API request.
- **📈 Automated Hyperparameter Optimization (HPO)**: A scheduled Celery Beat task runs nightly **Optuna** studies to find the Pareto-optimal frontier between inference latency and detection accuracy.
- **📊 Production Monitoring**: Out-of-the-box integration with **Prometheus** for RED (Rate, Errors, Duration) metrics and **Weights & Biases (W&B)** for ML experiment tracking.

---

## 🏗️ Architecture Diagram

The system is separated into highly cohesive microservices communicating via Redis message brokering:

1. **FastAPI (`api`)**: Handles incoming HTTP requests, serves Prometheus metrics, and manages the synchronous entry points.
2. **Celery Worker (`worker`)**: Consumes video analysis tasks from the Redis queue. Loads the heavy ML models into VRAM and processes the compute-intensive pipelines.
3. **Celery Beat (`beat`)**: The cron-scheduler that triggers the nightly Optuna HPO loops.
4. **Redis (`redis`)**: Acts as both the message broker for Celery and the fast KV-store for the zero-downtime configuration hot-swapper.
5. **Optuna Dashboard (`dashboard`)**: Visualizes the multi-objective optimization studies.

---

## 🚀 Getting Started

### Prerequisites
- Docker Engine & Docker Compose (v2+)
- NVIDIA GPU with CUDA Toolkit (Highly Recommended for inference)
- Windows / Linux / macOS

### Environment Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/signalsense.git
   cd signalsense
   ```

2. Configure environment variables. Create a `.env` file (or export them):
   ```env
   WANDB_API_KEY=your_wandb_api_key_here
   API_KEYS=your_secret_api_key
   REDIS_HOST=redis
   ```

### Running with Docker (Recommended)

SignalSense is optimized to run as a multi-container Docker application. The `docker-compose.yml` file is explicitly configured to deduplicate massive PyTorch builds, utilizing BuildKit caching.

Run the entire stack:
```bash
docker compose up --build -d
```

### Services Deployed
- **REST API**: `http://localhost:8000`
- **API Documentation (Swagger UI)**: `http://localhost:8000/docs`
- **Optuna Dashboard**: `http://localhost:8080`
- **Prometheus Metrics**: `http://localhost:8000/metrics`

---

## 💻 Local Development (Without Docker)

If you prefer to run the system natively (e.g., for debugging):

1. **Start Redis**:
   ```bash
   docker run -d -p 6379:6379 --name signalsense-redis redis
   ```
2. **Setup Python Virtual Environment** (Requires Python 3.11):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
   ```
3. **Start the API**:
   ```bash
   uvicorn serving.main:app --reload
   ```
4. **Start the Celery Worker**:
   ```bash
   celery -A serving.tasks worker --loglevel=info --pool=solo
   ```

---

## 📂 Project Structure

```text
signalsense/
├── cv_pipeline/            # Core Vision Models (YOLO, CLIP, DINOv2)
│   ├── clip_classifier.py  # Zero-shot scene classification
│   ├── detector.py         # YOLOv9 object detection
│   └── temporal_model.py   # DINOv2 temporal sequence analysis
├── llm_pipeline/           # Multimodal LLM & Evaluation
│   ├── trainer.py          # QLoRA fine-tuning for Phi-3.5-Vision
│   └── evaluate.py         # BERTScore & semantic evaluation
├── hpo/                    # Hyperparameter Optimization
│   ├── hot_swap.py         # Zero-downtime Redis config sync
│   ├── objective.py        # Multi-objective Optuna evaluation
│   └── search_spaces.py    # Parameter boundaries for CV & LoRA
├── serving/                # FastAPI & Celery Integration
│   ├── main.py             # FastAPI application and routing
│   ├── tasks.py            # Async Celery worker tasks
│   └── schemas.py          # Pydantic validation models
├── monitoring/             # Observability
│   └── metrics.py          # Prometheus RED metrics
├── Dockerfile.api          # Unified Dockerfile with BuildKit caching
└── docker-compose.yml      # Multi-container deployment configuration
```

---

## ⚙️ How the Hot-Swapping Works

SignalSense uses a highly reliable Redis-backed configuration state. Instead of restarting the API or worker nodes when hyperparameters or model weights change:
1. The nightly Optuna study discovers a new Pareto-optimal configuration.
2. The `hpo/hot_swap.py` module pushes this JSON configuration to the `signalsense:active_config` Redis key.
3. The FastAPI lifespan manager and Celery workers dynamically poll this key and instantly switch active configurations in memory without dropping connections.

---

## 🛡️ License

This project is licensed under the MIT License - see the LICENSE file for details.
