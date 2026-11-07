from pathlib import Path
from unittest import result

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageEnhance, ImageOps
import pandas as pd
import torch
from fontTools.ufoLib.glifLib import validateLayerInfoVersion3ValueForAttribute
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


SCRIPT_DIR = Path(__file__).resolve().parent
NPU_LAB_DIR = SCRIPT_DIR.parent.parent

MODEL_FILE = NPU_LAB_DIR / "ch06" / "output" / "mobilenet_v2.onnx"
IMAGE_FILE  = NPU_LAB_DIR / "C04" / "C04_data" / "images" / "cat.jpg"

OUTPUT_DIR  = NPU_LAB_DIR / "ch08" / "output"


weights = MobileNet_V2_Weights.DEFAULT
model = mobilenet_v2(weights=weights)

model.eval()

preprocess = weights.transforms()

session = ort.InferenceSession(
    str(MODEL_FILE),
    providers=["CPUExecutionProvider"],
)

INPUT_NAME = session.get_inputs()[0].name
OUTPUT_NAME = session.get_outputs()[0].name

image = Image.open(IMAGE_FILE).convert("RGB")

test_inputs = {}

original_tensor = preprocess(image).unsqueeze(0)

test_inputs["original_image"] = original_tensor

flip_image = ImageOps.mirror(image)
flip_tensor = preprocess(flip_image).unsqueeze(0)
test_inputs["horizontal_flip"] = flip_tensor

dark_image = ImageEnhance.Brightness(image).enhance(0.5)
dark_tensor = preprocess(dark_image).unsqueeze(0)
test_inputs["brightness_50"] = dark_tensor

zero_tensor = torch.zeros(1,3,224,224, dtype=torch.float32)
test_inputs["zero_tensor"] = zero_tensor

torch.manual_seed(42)
random_tensor = torch.rand(1,3,224,224, dtype=torch.float32)
test_inputs["random_tensor"] = random_tensor

print()
print(f"검증 데이터 입력: {len(test_inputs)}")

def run_pytorch(input_tensor):
    with torch.inference_mode():
        output = model(input_tensor)


    return output.detach().cpu().numpy()

def run_onnx(input_tensor):
    input_numpy = input_tensor.detach().cpu().numpy().astype(np.float32)

    output = session.run(output_names = [OUTPUT_NAME], input_feed = {INPUT_NAME : input_numpy})[0]

    return output

RTOL = 1e-4
ATOL = 1e-4

validation_rows = []

for case_name, input_tensor in test_inputs.items():
    print()
    print('=' * 100)
    print(f"Test case: {case_name}")

    pytorch_output = run_pytorch(input_tensor)
    onnx_output = run_onnx(input_tensor)

    shape_match = pytorch_output.shape == onnx_output.shape
    abs_error = np.abs(pytorch_output - onnx_output)
    max_error = np.max(abs_error)
    mean_error = np.mean(abs_error)
    allclose = np.allclose(pytorch_output,
                           onnx_output,
                           rtol=RTOL,
                           atol=ATOL,
                           )

    pytorch_top1 = int(np.argmax(pytorch_output[0]))
    onnx_top1 = int(np.argmax(onnx_output[0]))

    top1_match = pytorch_top1 == onnx_top1

    passed = shape_match and allclose and top1_match

    validation_rows.append({
        "case_name": case_name,
        "shape_match": shape_match,
        "max_abs_error": max_error,
        "mean_error": mean_error,
        "allclose": allclose,
        "pytorch_top1": pytorch_top1,
        "onnx_top1": onnx_top1,
        "top1_match": top1_match,
        "result" : "PASS" if passed else "FAIL",
    })
    print(f"input shape : {tuple(input_tensor.shape)}")
    print(f"output shape : {pytorch_output.shape}")
    print(f"max error : {max_error:.10f}")
    print(f"Mean Error: {mean_error:.10f}")
    print(f"Allclose: {allclose}")
    print(f"Top-1 Match: {top1_match}")
    print(f"result : {'PASS' if passed else 'FAIL'}")
