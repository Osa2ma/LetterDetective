# Letter Detective 🕵️

An AI-powered handwriting recognition app that identifies handwritten letters (A-Z) using deep learning.

## Features

- **Dual Model Support**: Choose between CNN and MLP architectures
- **Drawing Canvas**: Draw letters directly in the browser
- **Image Upload**: Upload handwritten letter images
- **Top 3 Predictions**: See the model's top 3 guesses with confidence scores
- **Adversarial Testing**: Add noise to test model robustness
- **Explainable AI**: Saliency maps show where the AI focuses

## Tech Stack

- **Frontend**: Streamlit
- **Backend**: FastAPI (optional, for API-based deployment)
- **ML Framework**: PyTorch
- **Dataset**: EMNIST Letters

## Installation

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/letter-detective.git
cd letter-detective

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Option 1: Streamlit Only (Recommended)
```bash
streamlit run app.py
```

### Option 2: FastAPI Backend + Streamlit Frontend
```bash
# Terminal 1: Start backend
python main.py

# Terminal 2: Start frontend
streamlit run app.py
```

## Project Structure

```
├── app.py           # Streamlit frontend
├── main.py          # FastAPI backend (optional)
├── inference.py     # Prediction logic
├── model.py         # Model architectures (CNN, MLP)
├── train.py         # Training script
├── cnn_model.pth    # Trained CNN weights
├── mlp_model.pth    # Trained MLP weights
└── notebook.ipynb   # Training notebook
```

## Model Performance

- **CNN**: ~91% accuracy on EMNIST Letters
- **MLP**: ~85% accuracy on EMNIST Letters

## License

MIT
