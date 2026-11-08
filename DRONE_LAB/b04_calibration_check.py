from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18



SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_FILE = SCRIPT_DIR / "input" / "b02_resnet18_drone_best.pth"

CALIB_DIR = SCRIPT_DIR / "calib" / "drone"

OUTPUT_DIR = SCRIPT_DIR / "output"

image_files = []

for extension in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
    image_files.extend(CALIB_DIR.glob(extension))


image_files = sorted(set(image_files))

MAX_CALIB_IMAGES = 100

image_files = image_files[:MAX_CALIB_IMAGES]

checkpoint = torch.load(MODEL_FILE, map_location="cpu")

model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, checkpoint["num_classes"])
model.load_state_dict(checkpoint["state_dict"])

model.eval()


# b02 Validation Transform과 동일
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(checkpoint["image_size"]),
    transforms.ToTensor(),
    transforms.Normalize(mean=checkpoint["mean"], std=checkpoint["std"]),
])

target_layers = {
    "conv1": model.conv1,
    "layer1": model.layer1,
    "layer2": model.layer2,
    "layer3": model.layer3,
    "layer4": model.layer4,
}

for layer_name in target_layers:
    print(layer_name)


calibration_stats = {}

for layer_name in target_layers:
    calibration_stats[layer_name] = {
        "min" : float("inf"),
        "max" : float("-inf"),
        "sample_activation" : None
    }

def make_hook(layer_name):
    def hook(module, inputs, output):
        activation = output.detach().cpu().float()

        current_min  = activation.min().item()
        current_max = activation.max().item()

        calibration_stats[layer_name]["min"] = min(calibration_stats[layer_name]["min"], current_min)

        calibration_stats[layer_name]["max"] = max(calibration_stats[layer_name]["max"], current_max)

        if calibration_stats[layer_name]["sample_activation"] is None:
            calibration_stats[layer_name]["sample_activation"] = activation.clone()

    return hook


hooks = []

for layer_name, layer in target_layers.items():
    handle = layer.register_forward_hook(make_hook(layer_name))
    hooks.append(handle)

print("\n===== Calibration Start ======")

processed = 0

with torch.inference_mode():
    for index, image_file in enumerate(image_files, start=1):
        try:
            image = Image.open(image_file).convert("RGB")
        except Exception as e:
            print("[image error]", image_file.name, e)
            continue

        input_tensor = preprocess(image).unsqueeze(0)

        _ = model(input_tensor)

        processed += 1

        if processed % 10 == 0 or index == len(image_files):
            print(f"{index:3d} / {len(image_files):3d} 처리 완료")

print()
print("Calibration 이미지 처리:", processed)

for handle in hooks:
    handle.remove()


calibration_rows = []

for layer_name, stat in calibration_stats.items():
    min_value, max_value = stat["min"], stat["max"]

    max_abs = max(abs(min_value), abs(max_value))

    if max_abs == 0:
        scale = 1.0
    else:
        scale = max_abs / 127.0


    calibration_rows.append({
        "layer_name" : layer_name,
        "activation_min" : min_value,
        "activation_max" : max_value,
        "max_abs" : max_abs,
        "int8_scale" : scale,
    })


calibration_df = pd.DataFrame(data=calibration_rows)
print()
print("==== calibration range ====")
print(calibration_df.to_string(index=False))


quantization_rows = []

for layer_name, stat in calibration_stats.items():
    activation = stat["sample_activation"]
    if activation is None:
        continue
    row = calibration_df[calibration_df["layer_name"] == layer_name].iloc[0]
    scale = float(row["int8_scale"])

    quantized = torch.round(activation / scale)
    quantized = torch.clamp(quantized, -127, 127).to(torch.int8)

    restored = quantized.float() * scale

    error = torch.abs(activation - restored)
    max_error = error.max().item()
    mean_error = error.mean().item()

    saturation_count = ((quantized == -127) | (quantized == 127)).sum().item()

    total_count = quantized.numel()

    saturation_ratio = saturation_count / total_count * 100

    quantization_rows.append({
        "layer" : layer_name,
        "activation_elements": total_count,
        "scale": scale,
        "max_abs_error": max_error,
        "mean_abs_error": mean_error,
        "saturation_count": saturation_count,
        "saturation_ratio": saturation_ratio,
    })

quantization_df = pd.DataFrame.from_records(quantization_rows)

print("\n Activation Quantization")
print(quantization_df.to_string(index=False))


