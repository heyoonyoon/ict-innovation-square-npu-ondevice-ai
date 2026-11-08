from pathlib import Path
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Subset

from torchvision import datasets, transforms
from torchvision.models import (
    resnet18,
    ResNet18_Weights,
)


# ============================================================
# 1. 경로 설정
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent

DATA_DIR = (
    SCRIPT_DIR
    / "data"
)

OUTPUT_DIR = (
    SCRIPT_DIR
    / "output"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


BEST_MODEL_PATH = (
    OUTPUT_DIR
    / "b02_resnet18_drone_best.pth"
)

HISTORY_CSV_PATH = (
    OUTPUT_DIR
    / "b02_training_history.csv"
)

SAMPLE_CSV_PATH = (
    OUTPUT_DIR
    / "b02_sample_predictions.csv"
)


# ============================================================
# 2. 학습 설정
# ============================================================
SEED = 42

BATCH_SIZE = 32
EPOCHS = 5
LR = 0.001

TRAIN_RATIO = 0.8

CLASS_NAMES = [
    "bird",
    "drone",
    "sky",
]


# ============================================================
# 3. Seed 고정
# ============================================================
random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 4. Device 설정
# ============================================================
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print("=" * 70)
print("B02 - ResNet18 Drone/Bird/Sky Transfer Learning")
print("=" * 70)

print(
    f"[INFO] Device        : {DEVICE}"
)

print(
    f"[INFO] Batch Size    : {BATCH_SIZE}"
)

print(
    f"[INFO] Epochs        : {EPOCHS}"
)

print(
    f"[INFO] Learning Rate : {LR}"
)


# ============================================================
# 5. ImageNet 전처리 기준
# ============================================================
IMAGENET_MEAN = [
    0.485,
    0.456,
    0.406,
]

IMAGENET_STD = [
    0.229,
    0.224,
    0.225,
]


# ============================================================
# 6. Training Transform
# ============================================================
train_transform = transforms.Compose(
    [
        transforms.Resize(256),

        transforms.RandomResizedCrop(
            224,
            scale=(0.8, 1.0),
        ),

        transforms.RandomHorizontalFlip(
            p=0.5
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ]
)


# ============================================================
# 7. Validation Transform
# ============================================================
val_transform = transforms.Compose(
    [
        transforms.Resize(256),

        transforms.CenterCrop(224),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ]
)


# ============================================================
# 8. Drone Dataset 로드 (ImageFolder)
#
# data/drone_cls/{bird,drone,sky}
#
# 0 = bird
# 1 = drone
# 2 = sky
# ============================================================
print("\n[STEP 1] Drone Dataset Loading")


train_base_dataset = datasets.ImageFolder(
    root=DATA_DIR / "drone_cls",
    transform=train_transform,
)


val_base_dataset = datasets.ImageFolder(
    root=DATA_DIR / "drone_cls",
    transform=val_transform,
)


print(
    "[INFO] Total trainval images :",
    len(train_base_dataset),
)


# ============================================================
# 9. Train / Validation Index 생성
#
# 같은 원본 Dataset을
# 80% Train / 20% Validation으로 분리
# ============================================================
indices = np.arange(
    len(train_base_dataset)
)


rng = np.random.default_rng(
    SEED
)


rng.shuffle(
    indices
)


split_index = int(
    len(indices)
    * TRAIN_RATIO
)


train_indices = (
    indices[
        :split_index
    ].tolist()
)


val_indices = (
    indices[
        split_index:
    ].tolist()
)


print(
    "[INFO] Train images :",
    len(train_indices),
)

print(
    "[INFO] Val images   :",
    len(val_indices),
)


# ============================================================
# 10. Subset 생성
#
# Train은 train_transform
# Validation은 val_transform 사용
# ============================================================
train_dataset = Subset(
    train_base_dataset,
    train_indices,
)


val_dataset = Subset(
    val_base_dataset,
    val_indices,
)


# ============================================================
# 11. DataLoader
#
# Windows에서 multiprocessing 문제를 피하기 위해
# num_workers=0 사용
# ============================================================
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


print(
    "[INFO] Train batches :",
    len(train_loader),
)

print(
    "[INFO] Val batches   :",
    len(val_loader),
)


# ============================================================
# 12. ImageNet Pretrained ResNet18
# ============================================================
print("\n[STEP 2] ResNet18 Loading")


weights = ResNet18_Weights.DEFAULT


model = resnet18(
    weights=weights
)


print(
    "[INFO] Original FC :",
    model.fc
)


# ============================================================
# 13. Backbone Freeze
#
# ResNet18 전체 Parameter를 먼저 동결한다.
# ============================================================
for parameter in model.parameters():
    parameter.requires_grad = False


# ============================================================
# 14. Classification Head 교체
#
# 기존:
#
# 512 → 1000 ImageNet classes
#
# 변경:
#
# 512 → 2
#
# 0 = Cat
# 1 = Dog
# ============================================================
in_features = (
    model.fc.in_features
)


model.fc = nn.Linear(
    in_features,
    len(CLASS_NAMES),
)


print(
    "[INFO] New FC      :",
    model.fc
)


# ============================================================
# 15. 학습되는 Parameter 확인
# ============================================================
print("\n[INFO] Trainable Parameters")


for name, parameter in model.named_parameters():

    if parameter.requires_grad:

        print(
            " -",
            name,
            parameter.shape,
        )


# ============================================================
# 16. Device 이동
# ============================================================
model = model.to(
    DEVICE
)


# ============================================================
# 17. Loss Function
# ============================================================
criterion = (
    nn.CrossEntropyLoss()
)


# ============================================================
# 18. Optimizer
#
# Backbone은 동결했기 때문에
# FC Layer만 학습한다.
# ============================================================
optimizer = torch.optim.Adam(
    model.fc.parameters(),
    lr=LR,
)


# ============================================================
# 19. 학습 이력
# ============================================================
history = []


best_val_accuracy = -1.0


# ============================================================
# 20. Training Loop
# ============================================================
print("\n[STEP 3] Transfer Learning Start")


for epoch in range(
    1,
    EPOCHS + 1,
):

    print("\n" + "=" * 70)

    print(
        f"Epoch {epoch}/{EPOCHS}"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # 20-1. Training
    # --------------------------------------------------------
    model.train()


    train_loss_sum = 0.0

    train_correct = 0

    train_total = 0


    for images, labels in train_loader:

        images = images.to(
            DEVICE
        )

        labels = labels.to(
            DEVICE
        )


        # Gradient 초기화
        optimizer.zero_grad()


        # Forward
        logits = model(
            images
        )


        # Loss
        loss = criterion(
            logits,
            labels
        )


        # Backward
        loss.backward()


        # FC Weight Update
        optimizer.step()


        # -----------------------------------------------
        # 통계
        # -----------------------------------------------
        batch_size = (
            labels.size(0)
        )


        train_loss_sum += (
            loss.item()
            * batch_size
        )


        predictions = (
            logits.argmax(
                dim=1
            )
        )


        train_correct += (
            predictions
            ==
            labels
        ).sum().item()


        train_total += (
            batch_size
        )


    train_loss = (
        train_loss_sum
        /
        train_total
    )


    train_accuracy = (
        train_correct
        /
        train_total
    )


    # --------------------------------------------------------
    # 20-2. Validation
    # --------------------------------------------------------
    model.eval()


    val_loss_sum = 0.0

    val_correct = 0

    val_total = 0


    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(
                DEVICE
            )

            labels = labels.to(
                DEVICE
            )


            logits = model(
                images
            )


            loss = criterion(
                logits,
                labels
            )


            batch_size = (
                labels.size(0)
            )


            val_loss_sum += (
                loss.item()
                * batch_size
            )


            predictions = (
                logits.argmax(
                    dim=1
                )
            )


            val_correct += (
                predictions
                ==
                labels
            ).sum().item()


            val_total += (
                batch_size
            )


    val_loss = (
        val_loss_sum
        /
        val_total
    )


    val_accuracy = (
        val_correct
        /
        val_total
    )


    # --------------------------------------------------------
    # 20-3. Epoch 결과
    # --------------------------------------------------------
    print(
        f"Train Loss     : "
        f"{train_loss:.4f}"
    )

    print(
        f"Train Accuracy : "
        f"{train_accuracy:.4f}"
        f" ({train_accuracy * 100:.2f}%)"
    )

    print(
        f"Val Loss       : "
        f"{val_loss:.4f}"
    )

    print(
        f"Val Accuracy   : "
        f"{val_accuracy:.4f}"
        f" ({val_accuracy * 100:.2f}%)"
    )


    # --------------------------------------------------------
    # 20-4. History 저장
    # --------------------------------------------------------
    epoch_result = {
        "epoch":
            epoch,

        "train_loss":
            train_loss,

        "train_accuracy":
            train_accuracy,

        "val_loss":
            val_loss,

        "val_accuracy":
            val_accuracy,
    }


    history.append(
        epoch_result
    )


    # --------------------------------------------------------
    # 20-5. Best Model 저장
    #
    # 마지막 Epoch가 아니라
    # Validation Accuracy가 가장 좋은 모델을 저장한다.
    # --------------------------------------------------------
    if (
        val_accuracy
        >
        best_val_accuracy
    ):

        best_val_accuracy = (
            val_accuracy
        )


        checkpoint = {

            "model_name":
                "resnet18",

            "num_classes":
                len(CLASS_NAMES),

            "class_names":
                CLASS_NAMES,

            "state_dict":
                model.state_dict(),

            "best_val_accuracy":
                best_val_accuracy,

            "image_size":
                224,

            "mean":
                IMAGENET_MEAN,

            "std":
                IMAGENET_STD,

            "seed":
                SEED,
        }


        torch.save(
            checkpoint,
            BEST_MODEL_PATH,
        )


        print(
            "[BEST MODEL SAVED]"
        )

        print(
            BEST_MODEL_PATH
        )


# ============================================================
# 21. Training History CSV 저장
# ============================================================
history_df = pd.DataFrame(
    history
)


history_df.to_csv(
    HISTORY_CSV_PATH,
    index=False,
)


print("\n[STEP 4] Training Completed")

print(
    "[INFO] Best Validation Accuracy :",
    f"{best_val_accuracy * 100:.2f}%"
)

print(
    "[INFO] History CSV :",
    HISTORY_CSV_PATH
)

print(
    "[INFO] Best Model :",
    BEST_MODEL_PATH
)


# ============================================================
# 22. Best Model 다시 로드
#
# 저장이 제대로 되었는지 확인하고
# Sample Prediction을 수행한다.
# ============================================================
print("\n[STEP 5] Best Model Reload")


checkpoint = torch.load(
    BEST_MODEL_PATH,
    map_location=DEVICE,
)


best_model = resnet18(
    weights=None
)


best_model.fc = nn.Linear(
    best_model.fc.in_features,
    checkpoint[
        "num_classes"
    ],
)


best_model.load_state_dict(
    checkpoint[
        "state_dict"
    ]
)


best_model = best_model.to(
    DEVICE
)


best_model.eval()


# ============================================================
# 23. Validation Sample Prediction
#
# Validation Dataset의 앞 10개를 사용
# ============================================================
sample_results = []


print("\n[STEP 6] Sample Predictions")


with torch.no_grad():

    sample_count = min(
        10,
        len(val_dataset),
    )


    for sample_no in range(
        sample_count
    ):

        image_tensor, true_index = (
            val_dataset[
                sample_no
            ]
        )


        input_tensor = (
            image_tensor
            .unsqueeze(0)
            .to(DEVICE)
        )


        logits = best_model(
            input_tensor
        )


        probabilities = (
            torch.softmax(
                logits,
                dim=1,
            )
        )


        pred_index = int(
            probabilities.argmax(
                dim=1
            ).item()
        )


        confidence = float(
            probabilities[
                0,
                pred_index
            ].item()
        )


        true_index = int(
            true_index
        )


        true_class = (
            CLASS_NAMES[
                true_index
            ]
        )


        pred_class = (
            CLASS_NAMES[
                pred_index
            ]
        )


        correct = (
            true_index
            ==
            pred_index
        )


        dataset_index = (
            val_indices[
                sample_no
            ]
        )


        result = {

            "sample_no":
                sample_no + 1,

            "dataset_index":
                dataset_index,

            "true_index":
                true_index,

            "true_class":
                true_class,

            "pred_index":
                pred_index,

            "pred_class":
                pred_class,

            "confidence":
                confidence,

            "correct":
                correct,
        }


        sample_results.append(
            result
        )


        print(
            f"[{sample_no + 1:02d}] "
            f"True={true_class:<3} "
            f"Pred={pred_class:<3} "
            f"Confidence={confidence:.4f} "
            f"Correct={correct}"
        )


# ============================================================
# 24. Sample Prediction CSV 저장
# ============================================================
sample_df = pd.DataFrame(
    sample_results
)


sample_df.to_csv(
    SAMPLE_CSV_PATH,
    index=False,
)


print(
    "\n[INFO] Sample Prediction CSV :",
    SAMPLE_CSV_PATH
)


# ============================================================
# 25. 최종 출력
# ============================================================
print("\n" + "=" * 70)

print("B02 COMPLETE")

print("=" * 70)

print(
    "Best Model :",
    BEST_MODEL_PATH
)

print(
    "Best Val Accuracy :",
    f"{best_val_accuracy * 100:.2f}%"
)

print(
    "History CSV :",
    HISTORY_CSV_PATH
)

print(
    "Prediction CSV :",
    SAMPLE_CSV_PATH
)

print("=" * 70)