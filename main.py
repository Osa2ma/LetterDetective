from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from inference import get_model, predict, compute_saliency
import base64

# 1. INITIALIZE API
app = FastAPI(title="Handwriting Recognition API")

# 2. ENABLE CORS
# This allows your Streamlit frontend (running on port 8501) 
# to talk to this Backend (running on port 8000).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. LOAD MODELS ON STARTUP
print(" Loading models...")
models = {
    "cnn": get_model("cnn"),
    "mlp": get_model("mlp")
}

# Check what loaded successfully
loaded_names = [name for name, m in models.items() if m is not None]
print(f"Models ready: {loaded_names}")

@app.get("/")
def health_check():
    """
    Simple check to see if API is running.
    """
    return {
        "status": "online", 
        "available_models": loaded_names
    }

@app.post("/predict")
async def predict_digit(model_type: str = "cnn", noise: float = 0.0, file: UploadFile = File(...)):
    """
    Main prediction endpoint.
    - model_type: 'cnn' or 'mlp' (passed as query param)
    - noise: Adversarial noise level (0.0 to 1.0)
    - file: The image file
    """
    # 1. Select Model
    selected_model = models.get(model_type.lower())
    
    if selected_model is None:
        return {
            "error": f"Model '{model_type}' is not available.", 
            "available": loaded_names
        }

    # 2. Read Image
    image_bytes = await file.read()
    
    # 3. Predict
    # We use the clean 'predict' function from inference.py
    # Returns: [("A", 0.95), ("H", 0.03), ("R", 0.01)]
    top_3_results = predict(selected_model, image_bytes, noise_level=noise)
    
    # Top prediction
    top_letter, top_confidence = top_3_results[0]
    
    # Build top_3 list for frontend
    top_3_list = [{"letter": letter, "prob": prob} for letter, prob in top_3_results]
    
    return {
        "prediction": top_letter,
        "confidence": top_confidence,
        "top_3": top_3_list,
        "model_used": model_type,
        "noise_applied": noise
    }


@app.post("/explain")
async def explain_prediction(model_type: str = "cnn", file: UploadFile = File(...)):
    """
    Returns a saliency map heatmap showing where the model is looking.
    """
    selected_model = models.get(model_type.lower())
    
    if selected_model is None:
        return {"error": f"Model '{model_type}' is not available."}
    
    image_bytes = await file.read()
    
    # Compute saliency map
    heatmap_bytes = compute_saliency(selected_model, image_bytes)
    
    if heatmap_bytes is None:
        return {"error": "Failed to compute saliency map."}
    
    # Return as base64 encoded string
    heatmap_b64 = base64.b64encode(heatmap_bytes).decode('utf-8')
    
    return {
        "heatmap": heatmap_b64,
        "model_used": model_type
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)