import streamlit as st
from PIL import Image
import torch
import os
import time
import tempfile
from huggingface_hub import snapshot_download


class ImageGenerator:
    def __init__(self, ae_path, dit_path, qwen2vl_model_path, max_length=640):
        # Initialize the model with the provided paths
        self.ae_path = ae_path
        self.dit_path = dit_path
        self.qwen2vl_model_path = qwen2vl_model_path
        self.max_length = max_length

    def to_cuda(self):
        # Move model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load your model to the correct device
        pass


def inference(prompt, image, seed, size_level):
    # Add logic for your inference, model prediction, etc.
    return image, seed


# Set page config for better UI layout
st.set_page_config(page_title="Ghibli style", layout="centered")
st.title("🖼️ Ghibli style for Free : AI Image Editing")
st.markdown("Ghibli style images with AI.")

# === User Inputs ===
prompt = "Turn into an illustration in Studio Ghibli style"
uploaded_image = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])
seed = st.number_input("🎲 Random Seed (-1 for random)", value=-1, step=1)
size_level = st.number_input("📐 Size Level (minimum 512)", value=512, min_value=512, step=32)

generate_button = st.button("🚀 Generate")

# === Load Model (Cached) ===
@st.cache_resource
def load_model():
    repo = "stepfun-ai/Step1X-Edit"
    local_dir = "./step1x_weights"
    os.makedirs(local_dir, exist_ok=True)
    snapshot_download(repo_id=repo, local_dir=local_dir, local_dir_use_symlinks=False)

    model = ImageGenerator(
        ae_path=os.path.join(local_dir, 'vae.safetensors'),
        dit_path=os.path.join(local_dir, "step1x-edit-i1258.safetensors"),
        qwen2vl_model_path='Qwen/Qwen2.5-VL-7B-Instruct',
        max_length=640
    )
    model.to_cuda()
    return model

image_edit_model = load_model()

# === Inference and Image Display ===
if generate_button and uploaded_image is not None:
    input_image = Image.open(uploaded_image).convert("RGB")
    # Resize image for faster inference (adjust to your model's requirements)
    input_image.thumbnail((size_level, size_level))
    
    with st.spinner("🔄 Generating edited image..."):
        start = time.time()
        try:
            result_image, used_seed = inference(prompt, input_image, seed, size_level)
            end = time.time()

            st.success(f"✅ Done in {end - start:.2f} seconds — Seed used: {used_seed}")
            
            # Save and display the result in temporary file
            with tempfile.NamedTemporaryFile(dir="/tmp", delete=False, suffix=".png") as temp_file:
                result_image.save(temp_file.name)
                st.image(temp_file.name, caption="🖼️ Edited Image", use_column_width=True)
        except Exception as e:
            st.error(f"❌ Inference failed: {e}")
            st.stop()

