
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import ViltProcessor, ViltForQuestionAnswering
from pathlib import Path
import openvino
import nncf
import requests
from PIL import Image

MODEL_ID = "dandelin/vilt-b32-finetuned-vqa"
OPTIMIZED_MODEL_DIR = Path("optimized_model")
FP32_PATH = OPTIMIZED_MODEL_DIR / "vilt_vqa_fp32.xml"
FP16_PATH = OPTIMIZED_MODEL_DIR / "vilt_vqa_fp16.xml"
INT8_PATH = OPTIMIZED_MODEL_DIR / "vilt_vqa_int8.xml"

def main():
    """
    This is a one-time script to download, convert, and optimize all models.
    """
    if INT8_PATH.exists() and FP16_PATH.exists() and FP32_PATH.exists():
        print("✅ All models are already prepared. You can now run the Streamlit app.")
        return

    print("--- Starting One-Time Model Preparation ---")
    OPTIMIZED_MODEL_DIR.mkdir(exist_ok=True)
    
    print("➡️ [1/5] Downloading base model from Hugging Face...")
    processor = ViltProcessor.from_pretrained(MODEL_ID)
    pytorch_model = ViltForQuestionAnswering.from_pretrained(MODEL_ID)
    print("✅ [1/5] Download complete.")
    
    encoding = processor(images=torch.zeros(1, 3, 384, 384), text="dummy question", return_tensors="pt")
    example_input = {k: v for k, v in encoding.items()}

    print("➡️ [2/5] Converting PyTorch model to OpenVINO FP32...")
    ov_model_fp32 = openvino.convert_model(pytorch_model, example_input=example_input)
    openvino.save_model(ov_model_fp32, str(FP32_PATH))
    print("✅ [2/5] OpenVINO FP32 model saved.")

    print("➡️ [3/5] Compressing model to OpenVINO FP16...")
    openvino.save_model(ov_model_fp32, str(FP16_PATH), compress_to_fp16=True)
    print("✅ [3/5] OpenVINO FP16 model saved.")

    print("➡️ [4/5] Preparing calibration dataset for INT8 quantization...")
    def get_calibration_data(num_samples=10):
        image = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw).convert("RGB")
        questions = ["How many cats are there?", "What color is the couch?"] * 5
        for i in range(num_samples):
            encoding = processor(images=image, text=questions[i], return_tensors="pt")
                       yield {key: tensor for key, tensor in encoding.items()}
           

    calibration_dataset = nncf.Dataset(get_calibration_data())
    
    print("➡️ [5/5] Quantizing model to OpenVINO INT8 (this may take a minute)...")
    int8_model = nncf.quantize(ov_model_fp32, calibration_dataset, preset=nncf.QuantizationPreset.MIXED, subset_size=10)
    openvino.save_model(int8_model, str(INT8_PATH))
    print("✅ [5/5] OpenVINO INT8 model saved.")
    
    print("\n🎉 --- All models have been prepared successfully! --- 🎉")
    print("You can now run the Streamlit app with: streamlit run app.py")

if __name__ == "__main__":
    main()
