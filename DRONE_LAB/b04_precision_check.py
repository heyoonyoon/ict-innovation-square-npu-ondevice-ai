from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision.models import resnet18


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_FILE = SCRIPT_DIR / "input" / "b02_resnet18_drone_best.pth"

OUTPUT_DIR = SCRIPT_DIR / "output"

checkpoint = torch.load(MODEL_FILE, map_location="cpu")

model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, checkpoint["num_classes"])
model.load_state_dict(checkpoint["state_dict"])

model.eval()

# 전체 parameter 개수 확인

total_params = sum(parameter.numel() for parameter in model.parameters())
print("Total number of parameters: {:,}".format(total_params))


# 정밀도 별 저장 공간 계산

precision_info = [
    {
        "precision": "FP32",
        "bits": 32,
        "bytes_per_value": 4
    },
    {
        "precision": "FP16",
        "bits": 16,
        "bytes_per_value": 2
    },
    {
        "precision": "INT8",
        "bits": 8,
        "bytes_per_value": 1
    },
]

memory_rows = []

for item in precision_info:
    total_bytes = total_params * item["bytes_per_value"]

    total_mib = total_bytes / 1024 / 1024

    memory_rows.append({
        "precision": item["precision"],
        "bits": item["bits"],
        "bytes_per_value": item["bytes_per_value"],
        "parameter_count": total_params,
        "estimated_memory_MiB": total_mib,
    })

memory_df = pd.DataFrame(memory_rows)
print()
print("===== Precision Memory =====")
print(memory_df.to_string(index=False))

target_name = None
fp32_weight = None

for name, parameter in model.named_parameters():
    if parameter.ndim == 4:
        target_name = name
        fp32_weight = parameter.detach().cpu().clone().float()
        break


print(f"parameter: {target_name}")
print(f"shape: {fp32_weight.shape}")
print(f"dtype: {fp32_weight.dtype}")
print(f"element 수: {fp32_weight.numel()}")
print(f"최솟값: {fp32_weight.min().item()}")
print(f"최댓값: {fp32_weight.max().item()}")


fp32_bytes = fp32_weight.numel() * fp32_weight.element_size()

fp16_weight = fp32_weight.to(torch.float16)

fp16_bytes = fp16_weight.numel() * fp16_weight.element_size()

fp16_restored = fp16_weight.float()

fp16_error = torch.abs(fp32_weight - fp16_restored)
fp16_max_error = fp16_error.max().item()
fp16_mean_error = fp16_error.mean().item()

max_abs = fp32_weight.abs().max().item()

if max_abs == 0:
    scale = 1.0
else:
    scale = max_abs / 127.0

print()
print("===== Int8 Quantization =====")
print(f"max_abs: {max_abs:.10f}")
print(f"scale: {scale:.10f}")

int8_weight = torch.round(fp32_weight / scale)
int8_weight = torch.clamp(int8_weight, min=-127, max=127)

int8_weight = int8_weight.to(torch.int8)

int8_bytes = int8_weight.numel() * int8_weight.element_size()

int8_restored = int8_weight.float() * scale
int8_error = torch.abs(fp32_weight - int8_restored)
int8_max_error = int8_error.max().item()
int8_mean_error = int8_error.mean().item()

tensor_memory_df = pd.DataFrame([
    {
        "precision": "FP32",
        "dtype": str(fp32_weight.dtype),
        "element_size_byte": fp32_weight.element_size(),
        "memory_byte": fp32_bytes
    },
    {
        "precision": "FP16",
        "dtype": str(fp16_weight.dtype),
        "element_size_byte": fp16_weight.element_size(),
        "memory_byte": fp16_bytes
    },
    {
        "precision": "INT8",
        "dtype": str(int8_weight.dtype),
        "element_size_byte": int8_weight.element_size(),
        "memory_byte": int8_bytes
    }
])

print()
print("===== real tensor memory =====")
print(tensor_memory_df.to_string(index=False))

error_df = pd.DataFrame([
    {
        "precision": "FP32",
        "max_abs_error": 0.0,
        "mean_abs_error": 0.0,
    },
    {
        "precision": "FP16",
        "max_abs_error": fp16_max_error,
        "mean_abs_error": fp16_mean_error,
    },
    {
        "precision": "INT8",
        "max_abs_error": int8_max_error,
        "mean_abs_error": int8_mean_error,
    }
]
)

print()
print("===== numerical error =====")
print(error_df.to_string(index=False))






