from pathlib import Path

import csv
import statistics
import time

import torch
from PIL import Image

from torchvision.models import (
    mobilenet_v2,
    MobileNet_V2_Weights,
)


IMAGE_DIR = Path("../../C04/C04_data/images")


OUTPUT_FILE = Path("../output/05_image_profile.csv")


REPEAT = 10
WARMUP = 10


device = torch.device("cpu")
weights = MobileNet_V2_Weights.DEFAULT
preprocess = weights.transforms()
categories = weights.meta["categories"]

model = mobilenet_v2(weights=weights).to(device)
model.eval()


# ========================================
# Warm-up
# ========================================

sample = Image.open(
    IMAGE_DIR / "cat.jpg"
).convert("RGB")


sample_input = (
    preprocess(sample)
    .unsqueeze(0)
    .to(device)
)


with torch.inference_mode():
    for _ in range(WARMUP):
        _ = model(sample_input)

# ========================================
# 이미지별 측정
# ========================================

rows = []


for image_path in sorted(
    IMAGE_DIR.glob("*.jpg")
):

    end_to_end_times = []

    top1_class = None
    top1_score = None


    for _ in range(REPEAT):

        start = time.perf_counter()

        image = Image.open(
            image_path
        ).convert("RGB")


        input_batch = (
            preprocess(image)
            .unsqueeze(0)
            .to(device)
        )


        with torch.inference_mode():

            output = model(input_batch)


        probabilities = torch.softmax(
            output[0],
            dim=0,
        )


        score, index = torch.max(
            probabilities,
            dim=0,
        )


        end = time.perf_counter()


        end_to_end_times.append(
            (end - start) * 1000
        )


        top1_class = categories[index.item()]


        top1_score = score.item()


    rows.append({
        "file":image_path.name,

        "top1_class": top1_class,

        "top1_score": top1_score,

        "average_e2e_ms": statistics.mean( end_to_end_times),

        "p95_e2e_ms":
            sorted(end_to_end_times)[max(0, int(len(end_to_end_times) * 0.95) - 1)],
    })


# ========================================
# CSV 저장
# ========================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


with OUTPUT_FILE.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "file",
            "top1_class",
            "top1_score",
            "average_e2e_ms",
            "p95_e2e_ms",
        ],
    )


    writer.writeheader()

    writer.writerows(rows)


# ========================================
# 출력
# ========================================

for row in rows:

    print(
        f"{row['file']:20s} "
        f"{row['average_e2e_ms']:8.3f} ms "
        f"{row['top1_class']}"
    )


print()
print("CSV :",OUTPUT_FILE)