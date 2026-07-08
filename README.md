# 🎓 AI Classroom Assistant — Visual Question Answering with OpenVINO

Ask questions about any image and get instant AI-generated answers — then see how much faster it runs once optimized with Intel's OpenVINO toolkit.

This project uses [`dandelin/vilt-b32-finetuned-vqa`](https://huggingface.co/dandelin/vilt-b32-finetuned-vqa), a pre-trained ViLT (Vision-and-Language Transformer) model, to perform Visual Question Answering. A Streamlit web app lets you upload an image, ask a natural-language question, and compare inference **latency** and **model size** across the original PyTorch model and three OpenVINO-optimized variants — FP32, FP16, and INT8 — on whichever hardware devices (CPU, iGPU, etc.) are available on your machine.

---

## ✨ Features

- **Visual Question Answering** — Upload any image, ask a question in plain English, and get an AI-generated answer.
- **Interactive Web UI** — Clean, simple interface built with Streamlit.
- **Dynamic Hardware Selection** — Run inference on any OpenVINO-supported device detected on your system (e.g., CPU, integrated GPU).
- **Real-Time Performance Comparison** — Benchmark four configurations side by side:
  | Configuration | Description |
  |---|---|
  | Original PyTorch | Baseline, unoptimized FP32 model |
  | OpenVINO FP32 | Converted to OpenVINO IR, full precision |
  | OpenVINO FP16 | Half-precision — smaller and faster |
  | OpenVINO INT8 | 8-bit quantized — fastest and most compact |
- **Live Metrics** — Instantly view inference latency (ms) and on-disk model size (MB) for the selected configuration.

---

## 🧠 How It Works

The project is split into two stages so that model preparation (slow, one-time) is decoupled from the interactive app (fast, repeatable).

### Stage 1 — One-Time Model Preparation (`1_prepare_models.py`)

Run this once before using the app. It:

1. Downloads the base ViLT model and processor from the Hugging Face Hub.
2. Converts the PyTorch model to OpenVINO Intermediate Representation (IR) format (**FP32**).
3. Compresses the FP32 IR model into a half-precision **FP16** version.
4. Quantizes the FP32 model using a calibration dataset and the **NNCF** framework to produce an **INT8** version.

All optimized models are saved to the `optimized_model/` directory.

### Stage 2 — The Streamlit Application (`app.py`)

The interactive web app:

1. Loads the pre-optimized models produced in Stage 1 (no conversion happens at runtime, so startup is fast).
2. Lets you pick a hardware device and a model precision from the UI.
3. Compiles the chosen model for that target using the OpenVINO Runtime and runs inference on your image + question.
4. Displays the answer alongside latency and model size metrics.

---

## 📁 Project Structure

```
.
├── 1_prepare_models.py   # One-time setup: download, convert, and quantize models
├── app.py                # Streamlit web application
├── requirements.txt      # Python dependencies
├── optimized_model/      # Generated after running 1_prepare_models.py
│   ├── fp32/
│   ├── fp16/
│   └── int8/
└── README.md             # This file
```

---

## ⚙️ Prerequisites

- Python 3.9+
- A working internet connection (to download the base model on first run)
- (Optional) An Intel integrated GPU with appropriate drivers, if you want to benchmark GPU inference

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd <repository-folder>
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the models (run once)

```bash
python 1_prepare_models.py
```

This downloads the base model and generates the FP32, FP16, and INT8 OpenVINO models inside `optimized_model/`. This step can take several minutes depending on your internet speed and hardware.

### 5. Launch the app

```bash
streamlit run app.py
```

Streamlit will open the app in your browser (typically at `http://localhost:8501`).

---

## 🖥️ Usage

1. Upload an image using the file uploader.
2. Type a natural-language question about the image (e.g., *"What color is the car?"*).
3. Select a **hardware device** (CPU, GPU, etc.) from the dropdown.
4. Select a **model precision** (Original PyTorch, OpenVINO FP32, FP16, or INT8).
5. Click **Run Inference** to see the predicted answer along with latency and model size.
6. Try different device/precision combinations to compare performance directly.

---

## 📊 Why Compare Precisions?

Model optimization involves trade-offs between speed, size, and accuracy:

- **FP32 (OpenVINO)** — Same numerical precision as the original model, but benefits from OpenVINO's graph optimizations and hardware-specific acceleration.
- **FP16** — Roughly half the model size with minimal accuracy loss, and typically faster inference — especially on hardware with native FP16 support.
- **INT8** — The smallest and fastest option, using post-training quantization via NNCF. Best suited for latency-sensitive or resource-constrained deployments.

This app lets you see these trade-offs directly on your own hardware rather than relying on published benchmarks.

---

## 📦 Dependencies

All required packages are listed in `requirements.txt`, including:

- `streamlit`
- `torch` / `transformers`
- `openvino` / `openvino-dev`
- `nncf`
- `pillow`

Install them all with:

```bash
pip install -r requirements.txt
```

---

## 🛠️ Troubleshooting

- **No GPU option appears in the app** — Ensure Intel GPU drivers and the OpenVINO GPU plugin are installed, and that your GPU is supported by OpenVINO.
- **`optimized_model/` not found** — Make sure you've run `python 1_prepare_models.py` before launching `app.py`.
- **Slow first run** — The first execution downloads the base model from Hugging Face Hub; subsequent runs use the local cache.

---

## 🙏 Acknowledgments

- [ViLT: Vision-and-Language Transformer](https://huggingface.co/dandelin/vilt-b32-finetuned-vqa) by Dandelin
- [Intel OpenVINO Toolkit](https://github.com/openvinotoolkit/openvino)
- [NNCF (Neural Network Compression Framework)](https://github.com/openvinotoolkit/nncf)
- [Streamlit](https://streamlit.io/)

---

