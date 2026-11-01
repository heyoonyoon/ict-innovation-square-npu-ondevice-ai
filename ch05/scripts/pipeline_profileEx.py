from PIL import Image
import torch
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import time
from pathlib import Path


WARMUP = 10
ITERATIONS = 100
latencyies = []

device = torch.device('cpu')
weights = MobileNet_V2_Weights.DEFAULT
preprocess = weights.transforms()
model = mobilenet_v2(weights=weights).to(device)
model.eval()

image = Image.open("../../C04/C04_data/images/cat.jpg").convert('RGB')

IMAGE_PATH = Path("../../C04/C04_data/images/cat.jpg")
warm_input = preprocess(image).unsqueeze_(0).to(device)

with torch.inference_mode():
    for _ in range(WARMUP):
        _ = model(warm_input)


# Stage 결과 저장
load_times = []
preprocess_times = []
inference_times = []
postprocess_times = []
total_times = []

# pipeline benchmark

for _ in range(ITERATIONS):
    total_start = time.perf_counter()
    start = time.perf_counter()
    image = Image.open(IMAGE_PATH).convert('RGB')
    end = time.perf_counter()
    load_times.append(end - start)


    # preprocessing
    start = time.perf_counter()
    input_batch = preprocess(image).unsqueeze_(0).to(device)
    end = time.perf_counter()
    preprocess_times.append((end - start)*1000)


    # inference
    start = time.perf_counter()
    with torch.inference_mode():
        output = model(input_batch)
    end = time.perf_counter()
    inference_times.append((end - start)*1000)


    # postprocessing
    start = time.perf_counter()
    probabilities = torch.softmax(
        output[0],
        dim = 0
    )

    top5_prob, top5_idx = torch.topk(probabilities, 5)
    end = time.perf_counter()
    postprocess_times.append((end - start)*1000)

    total_end = time.perf_counter()
    total_times.append((total_end - total_start) * 1000)
    start = time.perf_counter()

import statistics

print("\n==== RESULTS =====")
print(f"Load time: {statistics.mean(load_times):.3f} ms")
print(f"Preprocess time: {statistics.mean(preprocess_times):.3f} ms")
print(f"Inference time: {statistics.mean(inference_times):.3f} ms")
print(f"Postprocess time: {statistics.mean(postprocess_times):.3f} ms")
print(f"Total time: {statistics.mean(total_times):.3f} ms")


