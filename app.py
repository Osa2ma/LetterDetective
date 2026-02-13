import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import io
import pandas as pd
import requests
import base64
import os

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Letter Detective",
    page_icon="🕵️",
    layout="wide"
)

# FastAPI Backend URL (use environment variable for deployment)
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# --- SIDEBAR ---
st.sidebar.header("⚙️ Settings")
st.sidebar.write("Choose the AI Brain:")
model_type = st.sidebar.radio(
    "Model Architecture", 
    ["CNN", "MLP"], 
    index=0,
    help="CNN is better for images. MLP is a simple linear network."
)

st.sidebar.markdown("---")
st.sidebar.subheader("Adversarial Testing")
noise_level = st.sidebar.slider(
    "Adversarial Noise", 
    0.0, 1.0, 0.0,
    help="Add random noise to test model robustness. Higher = more noise."
)

st.sidebar.markdown("---")
st.sidebar.subheader("Explainable AI")
show_saliency = st.sidebar.checkbox(
    "Show AI Attention (Saliency Map)",
    help="Visualize which pixels the model focuses on."
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**How to use:**\n"
    "1. Draw a letter on the canvas.\n"
    "2. (Optional) Upload an image file.\n"
    "3. Click 'Identify Letter'!"
)

# --- MAIN PAGE ---
st.title("🕵️ Letter Detective")
st.markdown(f"Using **{model_type}** model to identify handwritten letters.")
if noise_level > 0:
    st.warning(f"Adversarial noise enabled: {noise_level:.2f}")

col1, col2 = st.columns([1, 1])

# COLUMN 1: INPUT (Canvas)
with col1:
    st.subheader("1. Draw Here")
    # Canvas settings match EMNIST: Black stroke on White background
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 1)", 
        stroke_width=15,
        stroke_color="#000000", # Black Ink
        background_color="#FFFFFF", # White Paper
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )
    
    # Optional: File Uploader
    st.write("OR Upload an Image:")
    uploaded_file = st.file_uploader("Upload PNG/JPG", type=["png", "jpg", "jpeg"])

# COLUMN 2: RESULT
with col2:
    st.subheader("2. Prediction")
    
    if st.button("🔍 Identify Letter", type="primary"):
        image_bytes = None

        # Priority 1: Use Uploaded File
        if uploaded_file is not None:
            image_bytes = uploaded_file.getvalue()
            st.image(uploaded_file, caption="Uploaded Image", width=150)
            
        # Priority 2: Use Canvas Drawing
        elif canvas_result.image_data is not None:
            # Check if canvas is not empty (users often forget to draw)
            # We convert to PIL to check content easily
            img = Image.fromarray(canvas_result.image_data.astype("uint8"), "RGBA")
            
            # Convert to Bytes
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            image_bytes = buf.getvalue()
        
        # CALL FASTAPI BACKEND
        if image_bytes:
            try:
                with st.spinner(f"Asking the {model_type} model..."):
                    # Call /predict endpoint
                    files = {"file": ("image.png", image_bytes, "image/png")}
                    params = {"model_type": model_type.lower(), "noise": noise_level}
                    
                    response = requests.post(f"{API_URL}/predict", files=files, params=params)
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        pred_letter = result.get("prediction", "?")
                        confidence = result.get("confidence", 0.0)
                        top_3 = result.get("top_3", [])
                        
                        st.metric(label="Predicted Letter", value=pred_letter, delta=f"{confidence:.2%}")
                        
                        # --- CONFUSION CHART (Top 3) ---
                        if top_3:
                            st.subheader("Top 3 Predictions")
                            chart_data = pd.DataFrame({
                                "Letter": [item["letter"] for item in top_3],
                                "Probability": [item["prob"] for item in top_3]
                            }).set_index("Letter")
                            st.bar_chart(chart_data)
                        
                        if pred_letter == "?":
                            st.warning("The model is unsure.")
                        
                        # --- SALIENCY MAP (Explainable AI) ---
                        if show_saliency:
                            st.subheader("AI Attention Map")
                            explain_files = {"file": ("image.png", image_bytes, "image/png")}
                            explain_params = {"model_type": model_type.lower()}
                            
                            explain_response = requests.post(f"{API_URL}/explain", files=explain_files, params=explain_params)
                            
                            if explain_response.status_code == 200:
                                explain_result = explain_response.json()
                                heatmap_b64 = explain_result.get("heatmap")
                                
                                if heatmap_b64:
                                    heatmap_bytes = base64.b64decode(heatmap_b64)
                                    
                                    col_orig, col_heat = st.columns(2)
                                    with col_orig:
                                        st.image(image_bytes, caption="Original", width=140)
                                    with col_heat:
                                        st.image(heatmap_bytes, caption="Where AI is looking", width=140)
                            else:
                                st.warning("Could not generate saliency map.")
                    else:
                        st.error(f"API Error: {response.status_code}")
                        st.json(response.json())
                        
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to Backend API.")
                st.info("Make sure to run `python main.py` first!")
        else:
            st.warning("Please draw something or upload an image first.")

st.markdown("---")
st.markdown("Built with **FastAPI**, **Streamlit**, and **PyTorch**.")