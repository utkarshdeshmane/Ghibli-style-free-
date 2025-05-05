import streamlit as st
from PIL import Image
import torch
import os
import time
from huggingface_hub import snapshot_download


class ImageGenerator:
    def __init__(self, ae_path, dit_path, qwen2vl_model_path, max_length=640):
        pass

    def to_cuda(self):
        pass

def inference(prompt, image, seed, size_level):
    return image, 42

st.set_page_config(page_title="Step1X Edit", layout="centered")
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

if generate_button and uploaded_image is not None:
    input_image = Image.open(uploaded_image).convert("RGB")
    with st.spinner("🔄 Generating edited image..."):
        start = time.time()
        result_image, used_seed = inference(prompt, input_image, seed, size_level)
        end = time.time()

    st.success(f"✅ Done in {end - start:.2f} seconds — Seed used: {used_seed}")
    st.image(result_image, caption="🖼️ Edited Image", use_column_width=True)

# === Example Prompts & Demos ===
st.markdown("### 🔁 Try Example Prompts")

example_prompts = [
    ("examples/meme.jpg", "Turn into an illustration in Studio Ghibli style"),
    ("examples/celeb_meme.jpg", "Replace the gray blazer with a leather jacket"),
    ("examples/cookie.png", "Remove the cookie"),
    ("examples/poster_orig.jpg", "Replace 'lambs' with 'llamas'"),
]

for path, ex_prompt in example_prompts:
    cols = st.columns([1, 2])
    with cols[0]:
        st.image(path, caption="Input Example", use_column_width=True)
    with cols[1]:
        st.write(f"**Prompt:** {ex_prompt}")
        if st.button(f"Try: {ex_prompt}", key=path):
            img = Image.open(path).convert("RGB")
            with st.spinner("🧠 Generating example result..."):
                result_image, used_seed = inference(ex_prompt, img, -1, 512)
            st.image(result_image, caption="🎨 Output Example", use_column_width=True)
            st.write(f"Seed used: {used_seed}")
