import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import io
import os
import numpy as np
from model import LetterCNN, LetterMLP

# 1. SETUP DEVICE
# CPU is sufficient and safer for simple inference deployments
DEVICE = torch.device("cpu")

# 2. DEFINE LABELS
# Maps index 0-25 to letters A-Z
LABELS = {i: chr(65 + i) for i in range(26)}

# 3. MODEL DIRECTORY (cross-platform compatible)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")


def get_model(model_type="cnn"):
    """
    Loads the requested model architecture and weights.
    
    Args:
        model_type: 'cnn' or 'mlp'
    
    Returns:
        Loaded PyTorch model in eval mode, or None if loading fails.
    """
    try:
        # Select Model Architecture
        if model_type == "cnn":
            model = LetterCNN(num_classes=26)
            path = os.path.join(MODELS_DIR, "cnn_model.pth")
        elif model_type == "mlp":
            model = LetterMLP(num_classes=26)
            path = os.path.join(MODELS_DIR, "mlp_model.pth")
        else:
            print(f"[ERROR] Unknown model type: {model_type}")
            return None

        # Load Weights
        if os.path.exists(path):
            # map_location='cpu' ensures it loads even if trained on GPU
            model.load_state_dict(torch.load(path, map_location=DEVICE))
            model.to(DEVICE)
            model.eval() # Set to evaluation mode (disable Dropout/BatchNorm training)
            print(f"[INFO] {model_type.upper()} model loaded successfully from {path}")
            return model
        else:
            print(f"[ERROR] Weight file not found: {path}")
            return None

    except Exception as e:
        print(f"[ERROR] Failed to load {model_type} model: {e}")
        return None

def transform_image(image_bytes):
    """
    Preprocesses the raw image bytes to match EMNIST format.
    """
    # 1. Open image from bytes
    image = Image.open(io.BytesIO(image_bytes))
    
    # 2. Convert to Grayscale (L)
    image = image.convert('L')
    
    # 3. INVERT COLORS
    # Real world: Black ink on White paper.
    # EMNIST Dataset: White ink on Black paper.
    # We must invert the image so the model understands it.
    image = ImageOps.invert(image)

    # 4. Standard Transforms
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1722,), (0.3309,))
    ])
    
    # 5. Add Batch Dimension: (1, 1, 28, 28)
    return transform(image).unsqueeze(0)

def predict(model, image_bytes, noise_level=0.0):
    """
    The main function called by the API.
    Returns the predicted letter and confidence score.
    
    Args:
        model: The loaded PyTorch model
        image_bytes: Raw image bytes
        noise_level: Adversarial noise (0.0 to 1.0). Higher = more noise.
    """
    if model is None:
        return "?", 0.0

    tensor = transform_image(image_bytes).to(DEVICE)
    
    # Add adversarial noise if requested
    if noise_level > 0:
        noise = torch.randn_like(tensor) * noise_level
        tensor = tensor + noise
    
    with torch.no_grad():
        outputs = model(tensor)
        
        # Calculate probabilities to get confidence score
        probs = torch.nn.functional.softmax(outputs, dim=1)
        
        # Get Top 3 predictions
        top_probs, top_indices = torch.topk(probs, 3, dim=1)
        
    # Build Top 3 results list: [("A", 0.95), ("H", 0.03), ("R", 0.01)]
    top_3 = []
    for i in range(3):
        idx = top_indices[0, i].item()
        prob = top_probs[0, i].item()
        letter = LABELS.get(idx, "?")
        top_3.append((letter, prob))
    
    return top_3


def compute_saliency(model, image_bytes):
    """
    Compute a saliency map showing which pixels the model focuses on.
    Uses SmoothGrad for better visualization (averages gradients over noisy samples).
    Returns the heatmap as PNG image bytes.
    """
    if model is None:
        return None
    
    # Disable gradients on model parameters (only input needs gradients)
    for param in model.parameters():
        param.requires_grad = False
    
    # 1. Preprocess image
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert('L')
    image = ImageOps.invert(image)
    
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1722,), (0.3309,))
    ])
    
    base_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    # 2. SmoothGrad: Average gradients over multiple noisy samples
    num_samples = 25
    noise_level = 0.15
    accumulated_grads = torch.zeros_like(base_tensor)
    
    for _ in range(num_samples):
        # Add noise to input
        noisy_tensor = base_tensor + torch.randn_like(base_tensor) * noise_level
        noisy_tensor = noisy_tensor.clone().detach().requires_grad_(True)
        
        # Forward pass
        outputs = model(noisy_tensor)
        
        # Get predicted class score
        predicted_idx = outputs.argmax(dim=1).item()
        score = outputs[0, predicted_idx]
        
        # Backward pass
        model.zero_grad()
        score.backward()
        
        # Accumulate absolute gradients
        accumulated_grads += noisy_tensor.grad.data.abs()
    
    # Average the gradients
    saliency = accumulated_grads[0, 0].cpu().numpy() / num_samples  # (28, 28)
    
    # 3. Apply Gaussian smoothing to reduce noise
    from scipy.ndimage import gaussian_filter
    saliency = gaussian_filter(saliency, sigma=1.0)
    
    # 4. Normalize to 0-1
    saliency_min = saliency.min()
    saliency_max = saliency.max()
    if saliency_max - saliency_min > 0:
        saliency = (saliency - saliency_min) / (saliency_max - saliency_min)
    else:
        saliency = np.zeros_like(saliency)
    
    # 5. Create overlay: Original image with green attention highlight
    # Get original grayscale image (28x28)
    orig_gray = np.array(image.resize((28, 28))) / 255.0  # 0-1
    
    # Create RGB image from grayscale
    overlay = np.stack([orig_gray, orig_gray, orig_gray], axis=-1)  # (28, 28, 3)
    
    # Add green channel based on saliency (high attention = bright green)
    overlay[:, :, 1] = np.clip(orig_gray + saliency * 0.8, 0, 1)  # Boost green
    overlay[:, :, 0] = orig_gray * (1 - saliency * 0.5)  # Reduce red where attention
    overlay[:, :, 2] = orig_gray * (1 - saliency * 0.5)  # Reduce blue where attention
    
    # Convert to uint8
    overlay = (overlay * 255).astype(np.uint8)
    
    # Resize to larger size for display
    heatmap_img = Image.fromarray(overlay, mode='RGB')
    heatmap_img = heatmap_img.resize((280, 280), Image.NEAREST)
    
    # Convert to bytes
    buf = io.BytesIO()
    heatmap_img.save(buf, format='PNG')
    buf.seek(0)
    
    return buf.getvalue()