# 🔍 Letter Detective

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

A deep learning web application for real-time handwritten letter recognition using PyTorch CNNs trained on the EMNIST dataset. Features explainable AI with saliency maps and adversarial robustness testing.

## ✨ Features

- **Real-time Letter Recognition** — Draw letters directly on canvas or upload images
- **Dual Model Support** — Choose between CNN (~95% accuracy) or MLP (~90% accuracy)
- **Top-3 Predictions** — View confidence scores for the three most likely letters
- **Explainable AI** — SmoothGrad saliency maps show which pixels influence predictions
- **Adversarial Testing** — Add Gaussian noise to test model robustness
- **RESTful API** — FastAPI backend for integration with other applications

## 🏗️ Architecture

```
┌─────────────────┐     HTTP      ┌─────────────────┐
│   Streamlit     │ ──────────►  │    FastAPI      │
│   Frontend      │  /predict    │    Backend      │
│   (app.py)      │  /explain    │   (main.py)     │
└─────────────────┘              └────────┬────────┘
                                          │
                                          ▼
                               ┌─────────────────────┐
                               │   Inference Layer   │
                               │   (inference.py)    │
                               └────────┬────────────┘
                                        │
                          ┌─────────────┴─────────────┐
                          ▼                           ▼
                   ┌────────────┐              ┌────────────┐
                   │ LetterCNN  │              │ LetterMLP  │
                   │  (91%)     │              │  (85%)     │
                   └────────────┘              └────────────┘
```

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | Streamlit, streamlit-drawable-canvas |
| Backend | FastAPI, Uvicorn |
| ML Framework | PyTorch |
| Dataset | EMNIST Letters (26 classes: A-Z) |
| Explainability | SmoothGrad Saliency Maps |

## 📁 Project Structure

```
Letter Detective/
├── app.py              # Streamlit frontend
├── main.py             # FastAPI backend
├── inference.py        # Model loading & prediction
├── train.py            # Training script
├── model.py            # Neural network architectures
├── requirements.txt    # Python dependencies
├── models/
│   ├── cnn_model.pth   # Trained CNN weights
│   └── mlp_model.pth   # Trained MLP weights
├── notebooks/
│   └── notebook.ipynb  # Experimentation notebook
└── data/
    └── EMNIST/         # Dataset (auto-downloaded)
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/letter-detective.git
   cd letter-detective
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   .\venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

**Option 1: Full Stack (API + Frontend)**
```bash
# Terminal 1: Start FastAPI backend
uvicorn main:app --reload --port 8000

# Terminal 2: Start Streamlit frontend
streamlit run app.py
```

**Option 2: Direct Mode (Frontend only)**
```bash
# Set environment variable
set USE_DIRECT_INFERENCE=true  # Windows
export USE_DIRECT_INFERENCE=true  # Linux/Mac

# Run Streamlit
streamlit run app.py
```

## 📊 Model Performance

| Model | Test Accuracy | Parameters | Inference Time |
|-------|--------------|------------|----------------|
| LetterCNN | ~95% | ~1.2M | ~5ms |
| LetterMLP | ~90% | ~400K | ~2ms |

### CNN Architecture

```
Input (1×28×28) → Conv2D(32) → ReLU → MaxPool
                → Conv2D(64) → ReLU → MaxPool
                → Flatten → Linear(256) → ReLU → Dropout(0.5)
                → Linear(26) → Output
```

## 🔌 API Endpoints

### `POST /predict`

Predict the letter from an image.

**Request Body:**
```json
{
  "image": "base64_encoded_image",
  "model_type": "cnn",
  "noise_level": 0.0
}
```

**Response:**
```json
{
  "prediction": "A",
  "confidence": 0.95,
  "top_3": [
    {"letter": "A", "confidence": 0.95},
    {"letter": "H", "confidence": 0.03},
    {"letter": "X", "confidence": 0.01}
  ],
  "noise_applied": 0.0
}
```

### `POST /explain`

Generate saliency map for explainability.

**Request Body:**
```json
{
  "image": "base64_encoded_image",
  "model_type": "cnn"
}
```

**Response:**
```json
{
  "heatmap": "base64_encoded_heatmap"
}
```

## 🧪 Training Your Own Model

```bash
python train.py
```

The training script will:
- Download EMNIST Letters dataset automatically
- Train a CNN for the configured number of epochs
- Save the best model to `models/cnn_model.pth`

## 🔒 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `USE_DIRECT_INFERENCE` | Skip API, use direct inference | `false` |
| `API_URL` | FastAPI backend URL | `http://localhost:8000` |

## 📝 License

This project is licensed under the MIT License.

---

