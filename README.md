# SignalSense AI Production System

SignalSense AI is a production-grade, distributed computer vision and natural language processing system designed for real-time video understanding, spatial-temporal object tracking, semantic anomaly detection, and automated text narration. The platform is architected to run at scale in multi-GPU cluster environments, leveraging asynchronous task queues, message brokers, self-supervised spatial embeddings, and large multimodal models (LMMs).

The primary goal of SignalSense AI is to convert raw, unstructured video feeds into structured semantic events, metrics, and human-readable natural language summaries. Additionally, the system incorporates fully automated nightly hyperparameter optimization loops that dynamically refine model thresholds, weights, and fine-tuning parameters, hot-swapping the configurations live with zero API downtime.

---

## Technical Stack and Architectural Overview

SignalSense AI is structured as a decoupled microservices architecture designed to isolate heavy GPU model execution from light HTTP serving layers:

*   **REST API Layer (FastAPI)**: Serves as the high-throughput gateway. It parses input payloads, performs Pydantic validation, schedules asynchronous analytics tasks, exposes real-time Prometheus RED (Rate, Errors, Duration) metrics, and handles zero-downtime hot-swap configuration lookups.
*   **Distributed Task Processing (Celery & Redis)**: An asynchronous queuing pipeline. Video tasks are routed through a Redis broker to a pool of Celery workers. The tasks utilize dynamic worker pools (`--pool=solo` on Windows/CPU fallback, or concurrent gevent/prefork on GPU-accelerated environments) to process incoming video streams concurrently without blocking HTTP request-response cycles.
*   **Core Machine Learning Engines**:
    *   **YOLOv9 (Object Detection)**: Used for high-accuracy spatial frame extraction, localization, and classification of specific object classes.
    *   **CLIP (OpenAI)**: Evaluates semantic similarity between raw visual frames and structural scene classification categories via zero-shot learning.
    *   **DINOv2 (Meta AI)**: Extracts high-fidelity self-supervised spatial embeddings to track temporal changes and structural anomalies across consecutive frames.
    *   **Phi-3.5-Vision (Multimodal LLM)**: Fine-tuned locally via Parameter-Efficient Fine-Tuning (PEFT/QLoRA) to generate synthetic conversational datasets, perform question-answering on video contexts, and produce high-quality natural language video narrations.
*   **Hyperparameter Optimization Engine (Optuna & Weights & Biases)**: A scheduled scheduler (Celery Beat) triggers automated studies using Optuna to evaluate multi-objective metrics. Experiment metadata, parameters, and loss curves are synchronized to Weights & Biases (W&B) for centralized engineering visibility.

---

## Detailed Model Ingestion & Processing Pipeline

When a video is submitted for processing, it undergoes a multi-stage distributed computational pipeline:

```
[Client Request] 
      │
      ▼
┌──────────────┐      Schedules      ┌─────────────┐
│ FastAPI API  │ ──────────────────> │ Redis Queue │
└──────────────┘                     └─────────────┘
                                            │
                                            ▼  Consumes Task
                                     ┌─────────────┐
                                     │   Celery    │
                                     │   Worker    │
                                     └─────────────┘
                                            │
                                            ▼
                                ┌──────────────────────┐
                                │ Frame Extraction     │
                                └──────────────────────┘
                                            │
                                            ▼
                                ┌──────────────────────┐
                                │ Parallel ML Stage    │
                                │ ──────────────────── │
                                │ 1. YOLOv9 Detection  │
                                │ 2. CLIP Semantics    │
                                │ 3. DINOv2 Embeddings │
                                └──────────────────────┘
                                            │
                                            ▼
                                ┌──────────────────────┐
                                │ Phi-3.5-Vision LLM   │
                                │ Narration & Summary  │
                                └──────────────────────┘
```

### 1. Ingestion & Frame Extraction
Videos are ingested asynchronously. The system extracts target keyframes at a variable sampling rate configured dynamically by the configuration engine (e.g., 1 frame per second, or custom interval rates defined during HPO). The `FrameExtractor` pipeline decodes video containers using OpenCV and yields standardized RGB numpy arrays.

### 2. YOLOv9 Spatial Inference
Standardized keyframes are passed directly to YOLOv9. The model detects bounding boxes, bounding coordinates, confidence thresholds, and class IDs. The outputs are immediately measured by the `signalsense_detections_per_frame` Prometheus metric.

### 3. CLIP Semantic Scene Classification
Concurrently, the same keyframes are passed through `CLIPSceneClassifier`. The model computes raw image features using a Vision Transformer (ViT-L/14) and measures cosine similarity against pre-encoded text prompts representing typical surveillance anomalies (e.g., "normal traffic", "physical altercation", "unattended package"). A softmax layer with a temperature scale projects the visual features into semantic probability spaces.

### 4. DINOv2 Feature Representation
For temporal coherence, Meta's self-supervised DINOv2 backbone generates dense spatial embeddings of the frames. The system calculates cosine distance between successive frame embeddings. Spikes in embedding distance highlight sudden environmental changes or structural anomalies that bypass standard class detectors.

### 5. Phi-3.5-Vision Narrative Generation
The extracted detections, semantic classifications, and temporal embeddings are packaged into a cognitive context. This context is injected into a custom prompt for the Phi-3.5-Vision model. The LLM performs an autoregressive decoding pass to generate a natural language narrative of the entire video segment.

---

## Zero-Downtime Hot-Swapping Architecture

To ensure the production pipeline can adapt to new models, hyperparameter thresholds, and weights without drops in service availability, SignalSense AI uses a centralized Redis-backed state machine:

```
                  ┌──────────────────────────────────────────────┐
                  │          Optuna Optimization Loop            │
                  │  Determines optimal thresholds & parameters  │
                  └──────────────────────────────────────────────┘
                                         │
                                         ▼ Writes Config Update
                              ┌─────────────────────┐
                              │     Redis State     │
                              │ (signalsense:config)│
                              └─────────────────────┘
                                   │           │
          Reads Active Config      │           │      Reads Active Config
          On Request Ingestion     │           │      On Task Execution
                                   ▼           ▼
                         ┌─────────────┐   ┌─────────────┐
                         │ FastAPI API │   │   Celery    │
                         │   Worker    │   │   Worker    │
                         └─────────────┘   └─────────────┘
```

1. **Configuration Schema**: All variables, such as YOLO IOU and confidence thresholds, CLIP temperature parameters, and fine-tuned LoRA adapter paths, are declared inside a structured schema.
2. **State Store**: The production database coordinates through Redis using the `signalsense:active_config` key.
3. **In-Memory Refresh**: The FastAPI middleware and active Celery tasks resolve this state dynamically. During request processing, the system references the active hot-swappable configuration in memory.
4. **Thread Safety**: Updates to model weights and thresholds are completed atomically using lock mechanisms, preventing concurrent workers from reading partial configurations during hot reloads.

---

## Automated Nightly Hyperparameter Optimization (HPO)

To maximize accuracy while strictly bound by latency SLAs, SignalSense AI runs automated nightly Optuna trials.

### The Optimization Objective
The scheduler triggers a multi-objective study evaluating:
1. **Model Accuracy**: Harmonic mean of the F1-Score of object detections and BERTScore semantic alignment on LLM narrations.
2. **Hardware Latency**: The total ingestion-to-annotation time in milliseconds (strictly capped at 30,000ms per stream).

```text
Objective = Maximize(Accuracy) AND Minimize(Latency)
```

### Search Spaces
Optuna explores the multidimensional space across the following parameters:
*   `conf_threshold` (float: 0.15 to 0.90) - Controls YOLO object detection sensitivity.
*   `iou_threshold` (float: 0.20 to 0.85) - Adjusts Non-Maximum Suppression overlap threshold.
*   `clip_temperature` (float: 0.01 to 0.50) - Modifies the scaling distribution of zero-shot classification confidence.
*   `lora_rank` (int: 8, 16, 32, 64) - Determines the rank of the parameter updates for LLM fine-tuning.
*   `lora_alpha` (int: 16, 32, 64, 128) - Controls the scaling multiplier for PEFT.

All results are written dynamically to the Optuna dashboard and tracked in W&B. The best performing set of parameters on the Pareto-frontier is automatically pushed to the hot-swapper, refreshing the production environment immediately.

---

## REST API Documentation

### Analyze Video Asynchronously
Submits a video URL for spatial-temporal and semantic model processing.

*   **URL**: `/api/v1/analyze`
*   **Method**: `POST`
*   **Headers**:
    *   `X-API-Key`: `dev-key-123`
    *   `Content-Type`: `application/json`
*   **Request Body**:
    ```json
    {
      "video_url": "https://example.com/assets/surveillance_stream.mp4",
      "callback_url": "https://callback.mycompany.com/webhook",
      "hpo_override": false
    }
    ```
*   **Response Body**:
    ```json
    {
      "task_id": "8fa2b879-1c9f-4318-87ee-a10c2c36a445",
      "status": "QUEUED",
      "submitted_at": "2026-05-18T14:44:59.000Z"
    }
    ```

### Prometheus Metrics Endpoint
Exposes operational and ML accuracy metrics to Prometheus scrapers.

*   **URL**: `/metrics`
*   **Method**: `GET`
*   **Exposed Metrics**:
    *   `signalsense_requests_total`: Total count of requests handled by the FastAPI application.
    *   `signalsense_request_duration_ms`: Latency histogram of the request lifecycle.
    *   `signalsense_model_inference_ms`: Per-model (`model_name`) raw inference duration.
    *   `signalsense_detections_per_frame`: Histogram tracking spatial denseness of elements in feeds.

---

## Docker Deployment and Orchestration

SignalSense AI uses a multi-container deployment architecture. The build utilizes BuildKit caches to optimize dependency compilation, avoiding large multi-gigabyte pip compilation failures.

### Build and Launch Instructions

1.  Make sure you have BuildKit enabled in your Docker daemon configuration.
2.  Start the multi-container stack in the background:
    ```bash
    docker compose up --build -d
    ```
3.  Monitor the logs to verify worker connections:
    ```bash
    docker compose logs -f api worker
    ```

### Deduplicated Build Details
The `docker-compose.yml` service architecture binds `api`, `worker`, and `beat` to the same build artifact: `Dockerfile.api`. When built, Docker compiles the Python environment exactly once, saves it under the tag `signalsense-base:latest`, and instantly provisions the multi-process microservices. This avoids concurrent dependency collisions and drops storage requirements by 66%.
