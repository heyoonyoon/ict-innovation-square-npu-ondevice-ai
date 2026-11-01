from PIL import Image
import torch
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import time

WARMUP = 10
ITERATIONS = 10
latencyies = []

device = torch.device('cpu')
weights = MobileNet_V2_Weights.DEFAULT
model = mobilenet_v2(weights=weights).to(device)

model.eval()

image = Image.open("../../C04/C04_data/images/cat.jpg").convert('RGB')

input_batch = (
    weights.transforms()(image)
    .unsqueeze_(0)
    .to(device)
)

print("=== Warm-up ===")

for i in range(10):
    start = time.perf_counter()
    with torch.inference_mode():
        _ = model (input_batch)

    end = time.perf_counter()
    latency_ms = (end - start) * 1000

    print(f"{i + 1: 2d}회 : {latency_ms:.3f}")

    latencyies.append(latency_ms)

import statistics


average = statistics.mean(latencyies)
minimum = min(latencyies)
maximum = max(latencyies)
std_dev = statistics.stdev(latencyies)

print()
print("=== RESULT ===")
print(f'Count: {ITERATIONS}')
print(f'Average: {average:.3f}')
print(f'Minimum: {minimum:.3f}')
print(f'Maximum: {maximum:.3f}')
print(f'Standard Deviation: {std_dev:.3f}')