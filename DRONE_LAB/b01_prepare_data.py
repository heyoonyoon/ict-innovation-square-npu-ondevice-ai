from pathlib import Path
import io
import random

import pyarrow.parquet as pq
from PIL import Image


# ============================================================
# 1. 경로 설정
#
# 원본 : data/raw/*.parquet
#        (Hugging Face athkot/drone-vs-bird-synthetic-10k, CC BY 4.0)
# 결과 : data/drone_cls/{bird,drone,sky}/*.jpg
#        (b02_train_drone.py 의 ImageFolder 입력)
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent

RAW_DIR = SCRIPT_DIR / "data" / "raw"

OUT_DIR = SCRIPT_DIR / "data" / "drone_cls"


# ============================================================
# 2. 설정
#
# 원본 category : 0 = drone, 1 = bird, 2 = airplane, 3 = helicopter
# 분류 class    : drone, bird, sky (빈 하늘 배경)
# ============================================================
SEED = 42

CATEGORY_TO_CLASS = {
    0: "drone",
    1: "bird",
}

MIN_OBJECT_SIZE = 40       # 이보다 작은 물체는 잘라도 형체가 안 보여서 제외
CONTEXT_SCALE = 2.0        # 물체 주변 하늘까지 포함하도록 박스를 2배로 확장
MIN_CROP_SIZE = 96

SKY_MIN_SIZE = 160
SKY_MAX_SIZE = 320
SKY_MARGIN = 8

MAX_PER_CLASS = 2000       # 클래스별 최대 장수 (클래스 균형)
SAVE_SIZE = 256            # 저장 해상도 (b02 Resize(256)과 동일)


random.seed(SEED)


def square_window(cx, cy, side, width, height):
    side = min(side, width, height)

    x0 = min(max(cx - side / 2, 0), width - side)
    y0 = min(max(cy - side / 2, 0), height - side)

    return (x0, y0, x0 + side, y0 + side)


def center_inside(box, window):
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2

    return window[0] <= cx <= window[2] and window[1] <= cy <= window[3]


def overlaps(box, window, margin):
    x, y, w, h = box

    return not (
        x + w + margin < window[0]
        or x - margin > window[2]
        or y + h + margin < window[1]
        or y - margin > window[3]
    )


def save(image, window, class_name, name):
    crop = image.crop(tuple(int(v) for v in window))
    crop = crop.resize((SAVE_SIZE, SAVE_SIZE), Image.BILINEAR)
    crop.save(OUT_DIR / class_name / f"{name}.jpg", quality=92)


# ============================================================
# 3. 후보 수집
# ============================================================
candidates = {"drone": [], "bird": [], "sky": []}

parquet_files = sorted(RAW_DIR.glob("*.parquet"))

print("[INFO] Parquet files :", [p.name for p in parquet_files])

for parquet_file in parquet_files:

    table = pq.read_table(parquet_file)

    for row in table.to_pylist():

        image_id = row["image_id"]
        width, height = row["width"], row["height"]
        boxes = row["objects"]["bbox"]
        categories = row["objects"]["category"]

        # 3-1. drone / bird crop
        for k, (box, category) in enumerate(zip(boxes, categories)):

            if category not in CATEGORY_TO_CLASS:
                continue

            x, y, w, h = box

            if max(w, h) < MIN_OBJECT_SIZE:
                continue

            side = max(max(w, h) * CONTEXT_SCALE, MIN_CROP_SIZE)
            window = square_window(x + w / 2, y + h / 2, side, width, height)

            # 다른 종류의 물체가 같이 잘리면 라벨이 섞이므로 제외
            mixed = any(
                other_category != category and center_inside(other_box, window)
                for other_box, other_category in zip(boxes, categories)
            )

            if mixed:
                continue

            candidates[CATEGORY_TO_CLASS[category]].append(
                (parquet_file, image_id, window, f"{image_id}_{k}")
            )

        # 3-2. sky crop : 어떤 물체와도 겹치지 않는 영역 1개
        for _ in range(20):

            side = random.uniform(SKY_MIN_SIZE, SKY_MAX_SIZE)
            x0 = random.uniform(0, width - side)
            y0 = random.uniform(0, height - side)
            window = (x0, y0, x0 + side, y0 + side)

            if not any(overlaps(box, window, SKY_MARGIN) for box in boxes):
                candidates["sky"].append(
                    (parquet_file, image_id, window, f"{image_id}_sky")
                )
                break

for class_name, items in candidates.items():
    print(f"[INFO] {class_name:5s} candidates : {len(items)}")


# ============================================================
# 4. 클래스 균형 맞추기
# ============================================================
per_class = min(MAX_PER_CLASS, *(len(items) for items in candidates.values()))

selected = {}

for class_name, items in candidates.items():
    random.shuffle(items)
    selected[class_name] = items[:per_class]

print("[INFO] Images per class :", per_class)


# ============================================================
# 5. 저장
# ============================================================
for class_name in selected:
    (OUT_DIR / class_name).mkdir(parents=True, exist_ok=True)

jobs = {}

for class_name, items in selected.items():
    for parquet_file, image_id, window, name in items:
        jobs.setdefault(parquet_file, {}).setdefault(image_id, []).append(
            (window, class_name, name)
        )

for parquet_file, by_image in jobs.items():

    table = pq.read_table(parquet_file, columns=["image", "image_id"])

    for row in table.to_pylist():

        if row["image_id"] not in by_image:
            continue

        image = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")

        for window, class_name, name in by_image[row["image_id"]]:
            save(image, window, class_name, name)

for class_name in selected:
    count = len(list((OUT_DIR / class_name).glob("*.jpg")))
    print(f"[OK] {class_name:5s} saved : {count}")
