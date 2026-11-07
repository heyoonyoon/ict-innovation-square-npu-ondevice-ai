from pathlib import Path
import time

import numpy as np
import onnxruntime as ort
from PIL import Image
from torch.compiler import keep_tensor_guards_unsafe

from torchvision.models import MobileNet_V2_Weights


SCRIPT_DIR = Path(__file__).resolve().parent
NPU_LAB_DIR = SCRIPT_DIR.parent.parent

MODEL_FILE = NPU_LAB_DIR / "ch06" / "output" / "mobilenet_v2.onnx"
IMAGE_FILE  = NPU_LAB_DIR / "C04" / "C04_data" / "images" / "cat.jpg"

OUTPUT_DIR  = NPU_LAB_DIR / "ch08" / "output"

print("ONNX 모델:", MODEL_FILE)
print("입력 이미지:", IMAGE_FILE)


session = ort.InferenceSession(
    str(MODEL_FILE),
    providers=["CPUExecutionProvider"],
)

print("session 생성 완료")
print("실제 사용 provider: ", session.get_providers())


input_info = session.get_inputs()[0]
output_info = session.get_outputs()[0]


print("\n=== Model Input ===")
print(f"name = {input_info.name}")
print(f"type = {input_info.type}")
print(f"shape = {input_info.shape}")


weights = MobileNet_V2_Weights.DEFAULT
preprocess = weights.transforms()
print(preprocess)

categories = weights.meta["categories"]
print(categories)

image = Image.open(IMAGE_FILE).convert("RGB")

print("\n==== 원본 이미지 ====")
print(f"image shape : {image.size}")


# 이미지 전처리
input_tensor = preprocess(image)
print(input_tensor.shape)
print(input_tensor.shape)
print(input_tensor.dtype)

input_batch = input_tensor.unsqueeze(0)
print("batch size:",input_batch.shape)


# pytorch tensor -> 이미지

input_numpy = input_batch.numpy().astype(np.float32)

print(f"\n{input_numpy.shape}")
print(input_numpy.dtype)


INPUT_NAME = input_info.name
OUTPUT_NAME = output_info.name

outputs = session.run(
    output_names = [OUTPUT_NAME],
    input_feed ={
    INPUT_NAME: input_numpy
    }
)

logits = outputs[0]

print("\n==== ONNX 출력 ====")
print(f"output shape: {logits.shape}")
print(f"output ty[e: {logits.dtype}")


def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

probabilities = softmax(logits)

# print(logits)
# print()
# print(probabilities)

# top-5
top5_indices = np.argsort(probabilities[0])[::-1][:5]

print("\n\n")

for rank, class_index in enumerate(top5_indices, start=1):
    probability = probabilities[0][class_index]
    class_name = categories[class_index]

    print(f"Rank {rank} : {class_name:<25} : {probability*100:6.2f}")


print()

# warm up
WARMUP = 10

for _ in range(WARMUP):
    _ = session.run(output_names=[OUTPUT_NAME],
                    input_feed = {INPUT_NAME: input_numpy})

print(f"\nWarm-up: {WARMUP}회 완료")

# onnx runtime latency 추정

REPEAT = 100
latencies = []

for _ in range(REPEAT):
    start_time = time.perf_counter()
    _ = session.run(output_names=[OUTPUT_NAME],
                    input_feed={INPUT_NAME: input_numpy})
    end_time = time.perf_counter()
    latency_ms = (end_time - start_time) * 1000
    latencies.append(latency_ms)

latencies = np.array(latencies)
mean_latency = np.mean(latencies)
min_latency = np.min(latencies)
max_latency = np.max(latencies)
std_latency = np.std(latencies)

throughput = 1000.0 / mean_latency

print()
print("==== onnx runtime 성능 ====")
print(f"반복횟수 {REPEAT}")
print(f"mean latency : {mean_latency:.3f} ms")
print(f"min latency : {min_latency:.3f} ms")
print(f"max latency : {max_latency:.3f} ms")
print(f"std latency : {std_latency:.3f} ms")
print(f"throughput : {throughput:.2f} images/sec")