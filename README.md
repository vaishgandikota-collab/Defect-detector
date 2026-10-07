<<<<<<< HEAD
# Intelligent Visual Recognition Suite: Real-Time Manufacturing Defect Detection

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.15+](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)](https://tensorflow.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An industrial-grade computer vision suite designed for real-time automated surface defect detection, classification, and visual explainability using Deep Convolutional Neural Networks (CNNs), Two-Stage Transfer Learning, and Regularization Suites.

---

## 🌟 Key Highlights

- **Multi-Model Benchmark Suite**: First-principles Baseline CNN, Two-Stage Transfer Learning (`ResNet50V2`), and Edge-Optimized Lightweight Deployment Model (`MobileNetV3Small`).
- **Zero Data Leakage**: Isolated train, validation, and untouched test splits; data augmentation applied exclusively during training.
- **Formal Regularization Ablation**: 6-experiment empirical study evaluating Dropout, Batch Normalization, L2 Regularization, and Augmentation.
- **Automated Explainability**: Real-time **Grad-CAM** saliency maps highlighting the exact spatial regions influencing classification.
- **Strict Enterprise Light Theme**: Designed with an accessible, high-contrast light palette (`#F8FAFC`, `#FFFFFF`, `#2563EB`).
- **Production REST API**: FastAPI backend supporting single and batch image predictions with rigorous schema validation.
- **Interactive Web Dashboard**: Streamlit dashboard equipped with live KPIs, dataset health explorer, conveyor-belt simulation, and deployment trade-off analytics.

---

## 📁 Project Structure

```text
manufacturing-defect-detector/
├── app.py                     # Streamlit Light-Theme Dashboard
├── README.md                  # Comprehensive Documentation
├── requirements.txt           # Pinned Dependencies
├── Dockerfile                 # Multi-Stage Docker Container
├── docker-compose.yml         # UI & API Container Orchestration
├── configs/
│   └── config.yaml            # Central System Configuration
├── data/
│   ├── raw/
│   ├── processed/
│   ├── train/
│   ├── validation/
│   └── test/
├── src/
│   ├── data/                  # Validation, Preprocessing, Augmentation, Loader
│   ├── models/                # Baseline CNN, ResNet50V2, MobileNetV3Small
│   ├── training/              # Training loop, Callbacks, Optuna Tuning, Ablation
│   ├── evaluation/            # Metrics, Confusion Matrix, Diagnostics, Grad-CAM
│   └── inference/             # Real-time Predictor & Disposition Logic
├── api/
│   └── main.py                # FastAPI Application & REST Endpoints
├── scripts/
│   ├── prepare_dataset.py     # Dataset Organizer & Synthesizer
│   ├── validate_dataset.py    # Health & Integrity Auditor
│   ├── train_baseline.py      # Baseline CNN Trainer
│   ├── train_transfer.py      # Two-Stage Transfer Learning Trainer
│   ├── run_ablation.py        # 6-Experiment Ablation Runner
│   ├── run_tuning.py          # Bayesian Hyperparameter Search
│   └── evaluate.py            # Comprehensive Test Evaluation
├── tests/                     # Pytest Unit & Integration Suite
└── reports/                   # Figures, CSV reports, and Final Academic Report
```

---

## 🚀 Quickstart Guide

### 1. Installation

Clone or enter the project directory and install the dependencies:

```bash
cd manufacturing-defect-detector
pip install -r requirements.txt
```

### 2. Dataset Preparation & Validation

Initialize and audit the manufacturing dataset:

```bash
python scripts/prepare_dataset.py
python scripts/validate_dataset.py
```

### 3. Training Candidate Models & Evaluating

Run training and multi-criteria test evaluation:

```bash
# Train baseline model
python scripts/train_baseline.py

# Train transfer learning model (2-Stage Fine-Tuning)
python scripts/train_transfer.py

# Run full evaluation, benchmark comparison, and champion model selection
python scripts/evaluate.py
```

### 4. Launching the Interactive Dashboard

```bash
python -m streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 5. Launching the REST API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger API documentation is available at `http://localhost:8000/docs`.

---

## 🧪 Automated Testing

Run the automated pytest test suite:

```bash
pytest -q
```

---

## 🐳 Docker Deployment

To launch both the Streamlit UI and FastAPI backend with Docker Compose:

```bash
docker-compose up --build
```

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
=======
# Defect-detector
An industrial-grade computer vision suite for real-time manufacturing surface defect detection using Deep CNNs, Two-Stage Transfer Learning (ResNet50V2), Edge MobileNetV3, and Grad-CAM explainability with sub-20ms latency.
>>>>>>> 1c20bd925f1718d85aaf71abdcbc1da5ab7502eb
