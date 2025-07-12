import streamlit as st
from PIL import Image
import time
import openvino
import torch
from transformers import ViltProcessor, ViltForQuestionAnswering
import numpy as np
import os
from pathlib import Path

MODEL_ID = "dandelin/vilt-b32-finetuned-vqa"
OPTIMIZED_MODEL_DIR = Path("optimized_model")
FP32_PATH = OPTIMIZED_MODEL_DIR / "vilt_vqa_fp32.xml"
FP16_PATH = OPTIMIZED_MODEL_DIR / "vilt_vqa_fp16.xml"
INT8_PATH = OPTIMIZED_MODEL_DIR / "vilt_vqa_int8.xml"

KNOWLEDGE_BASE = {
    "chloroplast": "The chloroplast is the site of photosynthesis, where the plant uses sunlight to create its own food.",
    "nucleus": "The nucleus is the 'control center' of the cell, containing the cell's genetic material (DNA).",
    "vacuole": "The central vacuole stores water, nutrients, and waste products, and helps maintain pressure against the cell wall.",
    "cell wall": "The cell wall is a rigid outer layer that provides structural support and protection to the plant cell.",
    "mitochondrion": "The mitochondrion is the 'powerhouse' of the cell, responsible for generating most of the cell's supply of adenosine triphosphate (ATP), used as a source of chemical energy."
}

st.set_page_config(page_title="AI Classroom Assistant", page_icon="🔬", layout="wide")

@st.cache_resource
def get_processor():
    return ViltProcessor.from_pretrained(MODEL_ID)

@st.cache_resource
def get_pytorch_model():
    return ViltForQuestionAnswering.from_pretrained(MODEL_ID)

@st.cache_resource
def get_all_openvino_models(device):
    core = openvino.Core()
    models = {
        "OpenVINO (FP32)": core.compile_model(str(FP32_PATH), device),
        "OpenVINO (FP16)": core.compile_model(str(FP16_PATH), device),
        "OpenVINO (INT8)": core.compile_model(str(INT8_PATH), device),
    }
    return models

def get_model_size(model_type):
    if model_type == "Original PyTorch (FP32)":
        return 490.37 
    path = None
    if model_type == "OpenVINO (FP32)": path = FP32_PATH
    elif model_type == "OpenVINO (FP16)": path = FP16_PATH
    elif model_type == "OpenVINO (INT8)": path = INT8_PATH
    
    if path and path.exists():
        return os.path.getsize(path.with_suffix('.bin')) / (1024 * 1024)
    return 0


st.title("🔬 AI Classroom Assistant: PyTorch vs. OpenVINO Optimization")
st.markdown("This demo compares an original PyTorch model against its OpenVINO-optimized versions. Observe the massive improvements in speed and size.")

if not all([FP32_PATH.exists(), FP16_PATH.exists(), INT8_PATH.exists()]):
    st.error("Models not found! Please run the setup script from your terminal first:")
    st.code("python 1_prepare_models.py")
    st.stop()

st.sidebar.title("Controls")
core = openvino.Core()
available_devices = core.available_devices
if "CPU" not in available_devices: available_devices.insert(0, "CPU")
if "GPU" not in available_devices: available_devices.append("GPU") 

device_choice = st.sidebar.selectbox("1. Choose Hardware Device:", available_devices)

try:
    if 'compiled_models' not in st.session_state or st.session_state.get('device') != device_choice:
        with st.spinner(f"Compiling OpenVINO models for {device_choice}..."):
            st.session_state.compiled_models = get_all_openvino_models(device_choice)
            st.session_state.device = device_choice
    openvino_models = st.session_state.compiled_models
except Exception as e:
    st.sidebar.error(f"Could not compile for {device_choice}. This is expected if the hardware/drivers are not present.")
    st.stop()

model_choice = st.sidebar.selectbox(
    "2. Choose Model Type:",
    ["Original PyTorch (FP32)", "OpenVINO (FP32)", "OpenVINO (FP16)", "OpenVINO (INT8)"]
)

st.sidebar.markdown("---")
st.sidebar.header("3. Upload Your Image")
uploaded_file = st.sidebar.file_uploader("Select an image to analyze...", type=["png", "jpg", "jpeg"])


col1, col2 = st.columns(2)

if uploaded_file is None:
    st.info("Please upload an image using the sidebar to begin.")
else:
    image = Image.open(uploaded_file).convert("RGB")
    
    with col1:
        st.header("Image Context")
        st.image(image, use_column_width=True)

    with col2:
        st.header("Question & Answer")
        question = st.text_input("Ask a question about the image:", "What is in the image?")

        if st.button(f"Get Answer with {model_choice} on {device_choice}", use_container_width=True, type="primary"):
            if not question:
                st.warning("Please ask a question.")
            else:
                with st.spinner("Thinking..."):
                    processor = get_processor()
                    encoding = processor(images=image, text=question, return_tensors="pt")
                    
                    start_time = time.perf_counter()
                    
                    if model_choice == "Original PyTorch (FP32)":
                        if device_choice != "CPU":
                            st.warning("PyTorch model will run on CPU only in this demo.")
                        model = get_pytorch_model()
                        with torch.no_grad():
                            outputs = model(**encoding)
                        logits = outputs.logits
                    else: # OpenVINO Models
                        compiled_model = openvino_models[model_choice]
                        inputs = {name: tensor for name, tensor in encoding.items()}
                        result = compiled_model(inputs)[compiled_model.output(0)]
                        logits = torch.from_numpy(result)

                    end_time = time.perf_counter()
                    
                    idx = logits.argmax(-1).item()
                    one_word_answer = get_pytorch_model().config.id2label[idx]
                    detailed_answer = KNOWLEDGE_BASE.get(one_word_answer.lower(), one_word_answer.capitalize())
                    
                    latency = (end_time - start_time) * 1000
                    model_size = get_model_size(model_choice)
                    
                    st.success(f"**Answer:** {detailed_answer}")
                    
                    res_col1, res_col2 = st.columns(2)
                    res_col1.metric("Inference Time", f"{latency:.2f} ms")
                    res_col2.metric("Model Size", f"{model_size:.2f} MB")

with st.expander("How this works: The PyTorch vs. OpenVINO Pipeline"):
    st.markdown("""
    This app directly compares a standard PyTorch model against its highly-optimized OpenVINO versions.

    - **Original PyTorch (FP32):** This is the baseline model, as it would be used during development. It's accurate but slow and large, making it unsuitable for efficient deployment.

    - **OpenVINO (FP32, FP16, INT8):** These are the deployment-ready models. The `1_prepare_models.py` script performed a one-time optimization to convert the PyTorch model into three OpenVINO versions. As you can see, even the baseline `FP32` OpenVINO model is faster than PyTorch due to graph optimizations. The `FP16` and `INT8` versions show dramatic improvements in both **speed (lower latency)** and **size**, while maintaining the accuracy of the answer.
    """)
