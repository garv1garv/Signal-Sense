import os

html_content = """<html xmlns:o='urn:schemas-microsoft-com:office:office' xmlns:w='urn:schemas-microsoft-com:office:word' xmlns='http://www.w3.org/TR/REC-html40'>
<head>
    <meta charset="utf-8">
    <title>SignalSense Research Report</title>
    <style>
        body { font-family: 'Calibri', sans-serif; line-height: 1.6; margin: 40px; color: #333; }
        h1 { color: #2C3E50; text-align: center; font-size: 28pt; margin-bottom: 10px; }
        h2 { color: #2980B9; border-bottom: 2px solid #2980B9; padding-bottom: 5px; font-size: 20pt; page-break-before: always; }
        h3 { color: #34495E; font-size: 16pt; margin-top: 20px; }
        p { font-size: 12pt; text-align: justify; margin-bottom: 15px; }
        .subtitle { text-align: center; font-size: 16pt; color: #7F8C8D; margin-bottom: 50px; font-style: italic; }
        .toc { font-size: 12pt; }
        .code-block { background-color: #F4F6F7; padding: 10px; border-left: 4px solid #BDC3C7; font-family: 'Courier New', monospace; font-size: 10pt; }
        ul, ol { font-size: 12pt; margin-bottom: 15px; }
        li { margin-bottom: 8px; }
        .page-break { page-break-after: always; }
        .cover-page { height: 80vh; display: flex; flex-direction: column; justify-content: center; align-items: center; }
    </style>
</head>
<body>

<div class="cover-page">
    <h1>SignalSense: Comprehensive Implementation & Research Report</h1>
    <div class="subtitle">Real-time video understanding API with auto-adapting models</div>
    <br><br><br><br>
    <p style="text-align: center;"><strong>Prepared by:</strong> AI Architect</p>
    <p style="text-align: center;"><strong>Date:</strong> 2026</p>
</div>
<div class="page-break"></div>

<h2>Executive Summary</h2>
<p>The SignalSense project represents a state-of-the-art approach to automated video surveillance and analysis. Traditional video surveillance relies heavily on human monitoring or simplistic motion-detection algorithms that suffer from high false-positive rates. SignalSense aims to solve this by deploying a fully automated, AI-driven pipeline that not only detects objects but understands the temporal context of scenes and generates human-readable narrations of events in real-time.</p>
<p>This report details the architectural decisions, technological stack, and file-by-file thought process behind the SignalSense system. The core innovation of this platform lies in its <strong>Auto-Adapting Pipeline</strong>. By integrating Hyperparameter Optimization (HPO) via Optuna and zero-downtime hot-swapping via Redis, the system continually fine-tunes itself overnight, adapting to new data distributions without requiring manual engineering intervention. This ensures the system maintains high accuracy (AUROC) and low latency against strict Service Level Agreements (SLAs).</p>

<h2>1. Technologies & Techniques Used</h2>
<p>The system leverages a sophisticated stack of modern machine learning and backend technologies to achieve real-time, accurate video understanding:</p>
<ul>
    <li><strong>YOLOv9:</strong> Utilized for high-speed, high-accuracy object detection. It extracts bounding boxes and object classes from frames.</li>
    <li><strong>CLIP (Contrastive Language-Image Pretraining):</strong> Employed as a zero-shot scene classifier. It maps images and predefined text categories (e.g., "safety violation", "crowd gathering") into the same latent space to classify frames without needing explicit training data for every possible scene.</li>
    <li><strong>DINOv2 & Transformers:</strong> DINOv2 extracts dense, self-supervised visual features (CLS tokens). These are fed into a custom PyTorch temporal transformer to identify anomalies across a sliding window of time.</li>
    <li><strong>Phi-3.5-Vision & QLoRA:</strong> A lightweight Vision-Language Model (VLM) used to narrate events. QLoRA (Quantized Low-Rank Adaptation) is used to fine-tune this model locally in 4-bit precision, allowing it to mimic larger, more expensive models like GPT-4o.</li>
    <li><strong>Optuna (NSGA-II):</strong> The engine behind the autonomous hyperparameter optimization. The NSGA-II algorithm finds the Pareto frontier across multiple competing objectives (accuracy, narration quality, and latency).</li>
    <li><strong>FastAPI & Celery:</strong> FastAPI provides the high-throughput asynchronous API serving layer, while Celery manages background jobs like the nightly HPO training loops.</li>
    <li><strong>Redis:</strong> Serves dual purposes: a message broker for Celery and an in-memory key-value store for hot-swapping model configurations with zero downtime.</li>
</ul>

<h2>2. Computer Vision Pipeline (cv_pipeline/)</h2>
<p>The Computer Vision pipeline is responsible for the raw perception of the video stream. It breaks down video files and extracts meaningful semantic information from the pixels.</p>

<h3>2.1 Frame Extractor (frame_extractor.py)</h3>
<p><strong>Thought Process:</strong> Processing every single frame of a 30 FPS video is computationally wasteful, especially for surveillance where scenes often remain static for long periods. The <code>FrameExtractor</code> implements a target FPS subsampling mechanism, but crucially adds a <strong>Perceptual Hashing (pHash)</strong> deduplication layer. By computing a DCT-based hash of the frame, the system compares consecutive frames. If the Hamming distance is below a threshold, the frame is deemed a duplicate and discarded. This technique drastically reduces the compute load on downstream heavy models (YOLO, CLIP, DINOv2) while preserving all meaningful temporal changes.</p>

<h3>2.2 Object Detector (detector.py)</h3>
<p><strong>Thought Process:</strong> Object detection provides the spatial grounding for the system. The <code>YOLODetector</code> class wraps Ultralytics YOLOv9. The critical architectural decision here is exposing the <code>conf_threshold</code> and <code>iou_threshold</code> as tunable parameters rather than hardcoding them. Different deployment environments have different precision/recall requirements (e.g., a highly secure area might tolerate false positives to ensure no missed detections, while a busy street needs high precision). These thresholds become targets for the HPO engine to optimize.</p>

<h3>2.3 Zero-Shot Scene Classifier (clip_classifier.py)</h3>
<p><strong>Thought Process:</strong> Training a classifier for every possible surveillance event is impossible due to data scarcity. The <code>CLIPSceneClassifier</code> uses OpenAI's CLIP model for zero-shot classification against a dynamic list of text prompts (e.g., "person in restricted area"). To optimize inference, the text embeddings for the categories are pre-computed and cached during initialization. During runtime, only the image is passed through the vision encoder, and cosine similarity is computed against the cached text features. The softmax <code>temperature</code> parameter is exposed for HPO tuning to control the sharpness of the probability distribution.</p>

<h3>2.4 Dense Embedding & Temporal Modeling (dino_embedder.py & temporal_model.py)</h3>
<p><strong>Thought Process:</strong> YOLO and CLIP operate on isolated frames. To understand <em>events</em> (which happen over time), the system needs temporal context. The <code>DINOEmbedder</code> extracts rich, self-supervised features from each frame. These dense vectors are then fed into the <code>TemporalEventDetector</code>, a lightweight PyTorch Transformer Encoder. By operating over a sliding window of embeddings, the transformer's self-attention mechanism learns temporal dependencies (e.g., distinguishing between a person walking normally versus someone falling). The window size, dropout rate, and transformer dimensions are all HPO targets to prevent overfitting and optimize latency.</p>

<h2>3. LLM Fine-Tuning Pipeline (llm_pipeline/)</h2>
<p>This pipeline is responsible for giving the system a "voice," translating raw bounding boxes and scene probabilities into coherent, structured JSON narrations.</p>

<h3>3.1 Synthetic Dataset Builder (dataset_builder.py)</h3>
<p><strong>Thought Process:</strong> The biggest bottleneck in training VLMs is acquiring high-quality labeled data. The <code>dataset_builder.py</code> employs a <strong>Teacher-Student Knowledge Distillation</strong> approach. It extracts sliding windows of frames from raw videos and passes them to a large, powerful (but slow and expensive) "Teacher" model (like GPT-4o or a large local VLM). The Teacher is prompted to act as a surveillance analyst and output a JSON containing the Event, Severity, and Reasoning. This script automates the creation of thousands of high-quality training pairs, creating a custom dataset without human annotators.</p>

<h3>3.2 QLoRA Trainer (trainer.py)</h3>
<p><strong>Thought Process:</strong> We cannot run massive models like GPT-4o in real-time for every camera feed. The <code>trainer.py</code> script takes a smaller "Student" model (Phi-3.5-Vision) and fine-tunes it on the synthetic dataset generated in the previous step. Because fine-tuning a full VLM requires massive VRAM, the system utilizes <strong>QLoRA</strong> (Quantized Low-Rank Adaptation). The base model is frozen in 4-bit NormalFloat precision (NF4), and small trainable adapter matrices (LoRA) are injected into the attention and MLP layers. This allows fine-tuning on a single consumer GPU. The script exposes crucial hyperparameters (Learning Rate, Batch Size, LoRA Rank, LoRA Alpha) for Optuna to optimize.</p>

<h3>3.3 Inference Engine (inference.py)</h3>
<p><strong>Thought Process:</strong> The <code>NarrationEngine</code> is designed for production serving. It loads the 4-bit base model and the dynamically trained LoRA adapters. Crucially, it includes a <code>reload_adapter()</code> method. This allows the system to hot-swap a newly trained, better-performing LoRA adapter into VRAM on the fly, without needing to restart the entire FastAPI server or reload the massive base model weights.</p>

<h2>4. Hyperparameter Optimization & Hot-Swapping (hpo/)</h2>
<p>The HPO module is the "brain" that allows SignalSense to be self-improving and autonomous.</p>

<h3>4.1 Search Spaces & Multi-Objective Function (search_spaces.py & objective.py)</h3>
<p><strong>Thought Process:</strong> Optimizing an AI pipeline is rarely about maximizing a single metric. If we only maximize accuracy, the system might become too slow. If we only minimize latency, the system becomes inaccurate. The <code>objective.py</code> script defines a multi-objective evaluation using Optuna. It evaluates three competing metrics simultaneously:
<ol>
    <li><strong>AUROC:</strong> Area Under the Receiver Operating Characteristic curve for anomaly detection.</li>
    <li><strong>BERTScore F1:</strong> A semantic similarity metric comparing the local VLM's narration to the Teacher's ground truth.</li>
    <li><strong>Negative p95 Latency:</strong> Inference speed.</li>
</ol>
The search space defines the bounds for YOLO thresholds, CLIP temperature, and LoRA hyperparameters.</p>

<h3>4.2 The NSGA-II Study Runner (run_study.py)</h3>
<p><strong>Thought Process:</strong> Using the <code>NSGAIISampler</code> (Non-dominated Sorting Genetic Algorithm II), Optuna efficiently explores the search space to find the <strong>Pareto Frontier</strong>. The Pareto frontier represents the set of configurations where you cannot improve one objective (e.g., Accuracy) without sacrificing another (e.g., Latency). The results are logged to an SQLite database for visualization via Optuna Dashboard, and to Weights & Biases (W&B) for rigorous experiment tracking.</p>

<h3>4.3 Zero-Downtime Hot-Swap (hot_swap.py)</h3>
<p><strong>Thought Process:</strong> Once a nightly HPO study finishes, the system must deploy the winning configuration. The <code>push_best_config()</code> function analyzes the Pareto frontier, selects the configuration with the highest AUROC that still meets the predefined latency SLA (e.g., under 200ms), and serializes it to a Redis key (<code>signalsense:active_config</code>). Because Redis is an in-memory datastore, this operation is instantaneous.</p>

<h2>5. API Serving and Background Workers (serving/)</h2>
<p>This module exposes the pipeline to end-users and manages the asynchronous background tasks.</p>

<h3>5.1 FastAPI Main Application (main.py)</h3>
<p><strong>Thought Process:</strong> FastAPI was chosen for its high performance and native async support. The application uses the <code>@asynccontextmanager lifespan</code> hook to load the heavy ML models into GPU memory exactly once upon server startup. For each incoming request on the <code>/v1/analyze</code> endpoint:
<ul>
    <li>The active configuration is fetched from Redis.</li>
    <li>The video is streamed through the Frame Extractor.</li>
    <li>Synchronous CV operations (YOLO, CLIP) are executed.</li>
    <li>The heavy VLM generation is offloaded to a separate thread pool using <code>asyncio.to_thread</code> so the ASGI event loop remains unblocked, allowing the server to handle concurrent requests efficiently.</li>
</ul></p>

<h3>5.2 Celery Tasks (tasks.py)</h3>
<p><strong>Thought Process:</strong> Long-running tasks, such as the nightly HPO training loops or batch processing of massive video archives, cannot block the HTTP API. Celery, backed by Redis, is used for reliable task queuing. The <code>run_nightly_hpo</code> task is scheduled via Celery Beat (like a cron job) to run at 2 AM daily. It executes the Optuna study and triggers the Redis hot-swap entirely in the background.</p>

<h2>6. Infrastructure & Deployment (Docker)</h2>
<p><strong>Thought Process:</strong> Deploying complex ML applications with PyTorch, CUDA, and system dependencies (like FFmpeg for OpenCV) is notoriously difficult. The project uses a multi-container Docker Compose architecture:
<ul>
    <li><strong>API Container:</strong> Runs the FastAPI server. Exposes ports.</li>
    <li><strong>Worker Container:</strong> Runs the Celery worker for ML training and batch processing.</li>
    <li><strong>Beat Container:</strong> Runs the Celery scheduler.</li>
    <li><strong>Redis Container:</strong> The central nervous system for caching and messaging.</li>
    <li><strong>Dashboard Container:</strong> Hosts the Optuna Dashboard for visualizing the Pareto frontier.</li>
</ul>
Both the API and Worker containers define GPU reservations in the <code>docker-compose.yml</code> file, ensuring they have access to the NVIDIA hardware required for deep learning inference and training.</p>

<h2>7. Conclusion</h2>
<p>SignalSense represents a paradigm shift in how AI pipelines are deployed. By heavily decoupling the perception modules (CV) from the reasoning modules (LLM), and binding them together with a self-optimizing HPO loop, the system eliminates the need for constant manual tweaking. It dynamically adapts its confidence thresholds and narration adapters based on the latest data, ensuring optimal performance against strict SLAs in real-world, highly variable environments.</p>

</body>
</html>
"""

# Save as a .doc file (Word reads HTML seamlessly when saved with a .doc extension)
file_path = os.path.join(os.getcwd(), 'SignalSense_Research_Report.doc')
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"Report successfully generated at: {file_path}")
