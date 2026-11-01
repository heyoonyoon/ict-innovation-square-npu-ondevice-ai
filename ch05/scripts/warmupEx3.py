from PIL import Image
import torch
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import time
from pathlib import Path
import csv


WARMUP = 10
ITERATIONS = 100
latencyies = []
OUTPUT_FILE = Path("../output/c05_latency_100.csv")

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

for i in range(ITERATIONS):
    start = time.perf_counter()
    with torch.inference_mode():
        _ = model (input_batch)

    end = time.perf_counter()
    latency_ms = (end - start) * 1000

    print(f"{i + 1: 2d}회 : {latency_ms:.3f}")

    latencyies.append(latency_ms)



OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
with OUTPUT_FILE.open(
    mode="w",
    newline="",
    encoding="utf-8",
) as f:
    writer = csv.writer(f)
    writer.writerow({
        "iteration",
        "latency-ms",
    })

    for i, value in enumerate(latencyies, start=1):
        writer.writerow([
            i,
            f"{value:.6f}",
        ])


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


