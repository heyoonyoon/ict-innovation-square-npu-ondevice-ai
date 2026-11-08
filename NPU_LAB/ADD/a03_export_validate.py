from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

import onnx
import onnxruntime as ort

from torchvision import datasets, transforms
from torchvision.models import resnet18


# ============================================================
# 1. 경로 설정
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent

INPUT_DIR = (
    SCRIPT_DIR
    / "input"
)

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


# ============================================================
# 2. 입력 / 출력 파일
# ============================================================
CHECKPOINT_FILE = (
    INPUT_DIR
    / "a02_resnet18_catdog_best.pth"
)

ONNX_FILE = (
    OUTPUT_DIR
    / "a03_resnet18_catdog.onnx"
)

SAMPLE_IMAGE_FILE = (
    OUTPUT_DIR
    / "a03_validation_sample.jpg"
)

RESULT_CSV_FILE = (
    OUTPUT_DIR
    / "a03_conversion_validation.csv"
)


# ============================================================
# 3. 검증 설정
#
# A02에서 Train / Validation Split을 만들 때 사용한
# 동일한 SEED를 사용해야 한다.
# ============================================================

SEED = 42

TRAIN_RATIO = 0.8


# ============================================================
# 4. PyTorch ↔ ONNX 수치 비교 허용 오차
# ============================================================

RTOL = 1e-4
ATOL = 1e-4


# ============================================================
# 5. 시작 메시지
# ============================================================
print("=" * 70)

print("A03 - PyTorch -> ONNX Export")

print("      PyTorch vs ONNX Runtime Validation")

print("=" * 70)


# ============================================================
# 6. A02 Checkpoint 존재 여부 확인
# ============================================================
if not CHECKPOINT_FILE.exists():

    raise FileNotFoundError(
        "\n"
        "A02 checkpoint를 찾을 수 없습니다.\n"
        f"Expected path:\n"
        f"{CHECKPOINT_FILE}\n"
    )


print(
    "[INFO] Checkpoint :",
    CHECKPOINT_FILE
)


# ============================================================
# 7. A02 Checkpoint 로드
# ============================================================
print(
    "\n[STEP 1] "
    "Load A02 checkpoint"
)


checkpoint = torch.load(
    CHECKPOINT_FILE,
    map_location="cpu",
)


# ============================================================
# 8. Checkpoint 정보 확인
# ============================================================
model_name = checkpoint["model_name"]

num_classes = checkpoint["num_classes"]

class_names = checkpoint["class_names"]

image_size = checkpoint["image_size"]

mean = checkpoint["mean"]

std = checkpoint["std"]

best_val_accuracy = checkpoint["best_val_accuracy"]


print(
    "[INFO] Model Name       :",
    model_name
)

print(
    "[INFO] Num Classes      :",
    num_classes
)

print(
    "[INFO] Class Names      :",
    class_names
)

print(
    "[INFO] Image Size       :",
    image_size
)

print(
    "[INFO] Mean             :",
    mean
)

print(
    "[INFO] Std              :",
    std
)

print(
    "[INFO] Best Val Accuracy:",
    best_val_accuracy
)


# ============================================================
# 9. ResNet18 구조 복원
#
# A02에서는:
#
# pretrained ResNet18
#        ↓
# FC Layer를 1000 → 2로 교체
#
# 했으므로 A03에서도 동일한 구조를 먼저 만든다.
#
# 여기서는 weights=None
# 왜냐하면 실제 Weight는 A02 checkpoint에서 가져오기 때문이다.
# ============================================================
print(
    "\n[STEP 2] "
    "Restore ResNet18 model"
)

model = resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    num_classes,
)


##########################################################################




##########################################################################


# ============================================================
# 10. A02에서 학습한 Weight 로드
# ============================================================
##########################################################################

model.load_state_dict(checkpoint["state_dict"])

##########################################################################


# ============================================================
# 11. 추론 모드
# ============================================================
model.eval()


print(
    "[INFO] Restored FC :",
    model.fc
)


# ============================================================
# 12. Validation Transform
#
# 반드시 A02 Validation과 동일해야 한다.
#
# Resize(256)
# CenterCrop(224)
# ToTensor()
# Normalize(ImageNet Mean / Std)
#
# Training Transform의 RandomResizedCrop,
# RandomHorizontalFlip을 사용하면 안 된다.
# ============================================================
##########################################################################

validation_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]
)



##########################################################################


# ============================================================
# 13. 실제 Oxford-IIIT Pet Dataset 로드
#
# A02와 동일한 trainval Dataset을 사용한다.
#
# target:
#
# 0 = Cat
# 1 = Dog
# ============================================================
print(
    "\n[STEP 3] "
    "Load Oxford-IIIT Pet"
)

raw_dataset = datasets.OxfordIIITPet(
    root=DATA_DIR,
    split="trainval",
    target_types="binary-category",
    transform=None,
    download=True
)

print(f"[INFO] Raw dataset images : {len(raw_dataset)}",)


##########################################################################




##########################################################################



# ============================================================
# 14. A02와 동일한 Train / Validation Split 복원
#
# A02에서:
#
# np.random.default_rng(SEED)
#
# 로 전체 Index를 Shuffle했으므로
# 동일한 로직을 사용하면 동일한 Validation Index가 나온다.
# ============================================================
indices = np.arange(len(raw_dataset))


rng = np.random.default_rng(SEED)


rng.shuffle(indices)


split_index = int(
    len(indices)
    * TRAIN_RATIO
)


train_indices = indices[:split_index].tolist()


val_indices = indices[split_index:].tolist()


print(
    "[INFO] Train images :",
    len(train_indices)
)

print(
    "[INFO] Val images   :",
    len(val_indices)
)


# ============================================================
# 15. Validation Image 1장 선택
#
# Validation Dataset의 첫 번째 실제 이미지를 사용한다.
#
# 중요한 점:
#
# 합성 이미지 X
# Dummy Image X
# 실제 Oxford-IIIT Pet 사진 O
# ============================================================
sample_dataset_index = val_indices[0]


sample_image, true_index = raw_dataset[sample_dataset_index]


true_index = int(true_index)


true_class = class_names[true_index]


print("\n[INFO] Validation Sample")

print(
    "Dataset Index :",
    sample_dataset_index
)

print(
    "True Index    :",
    true_index
)

print(
    "True Class    :",
    true_class
)


# ============================================================
# 16. 실제 이미지 저장
#
# 어떤 이미지를 PyTorch / ONNX 검증에 사용했는지
# 눈으로 확인할 수 있도록 저장한다.
# ============================================================
sample_image.save(SAMPLE_IMAGE_FILE)


print(
    "[INFO] Sample Image :",
    SAMPLE_IMAGE_FILE
)


# ============================================================
# 17. 실제 이미지를 Tensor로 변환
#
# 결과 Shape:
#
# [3, 224, 224]
# ============================================================
input_tensor = (
    validation_transform(
        sample_image
    )
)


print(
    "\n[INFO] Tensor Shape before batch :",
    tuple(input_tensor.shape)
)


# ============================================================
# 18. Batch Dimension 추가
#
# [3, 224, 224]
#
#        ↓
#
# [1, 3, 224, 224]
#
# NCHW
# ============================================================
input_tensor = (
    input_tensor
    .unsqueeze(0)
)


print(
    "[INFO] Tensor Shape after batch  :",
    tuple(
        input_tensor.shape
    )
)


print(
    "[INFO] Tensor dtype              :",
    input_tensor.dtype
)


print(
    "[INFO] Tensor min                :",
    float(input_tensor.min())
)


print(
    "[INFO] Tensor max                :",
    float(input_tensor.max())
)


# ============================================================
# 19. PyTorch FP32 추론
#
# 여기서 얻은 Output을
# ONNX Runtime 결과와 비교한다.
#
# Input 단위:
#
# Batch 1
#
# Shape:
#
# [1, 3, 224, 224]
# ============================================================
print(
    "\n[STEP 4] "
    "PyTorch inference"
)


with torch.no_grad():

    pytorch_output = model(input_tensor)


# NumPy 변환
pytorch_output_np = (
    pytorch_output
    .cpu()
    .numpy()
)


print(
    "[INFO] PyTorch Output Shape :",
    pytorch_output_np.shape
)


print("[INFO] PyTorch Logits :")

print(pytorch_output_np)


# ============================================================
# 20. PyTorch Top-1
# ============================================================
pytorch_top1_index = int(

    np.argmax(
        pytorch_output_np,
        axis=1
    )[0]
)


pytorch_top1_class = class_names[pytorch_top1_index]



print(
    "[INFO] PyTorch Top-1 :",
    pytorch_top1_class
)


# ============================================================
# 21. ONNX Export
#
# Hailo 변환을 고려하여 Static Shape로 만든다.
#
# Input:
#
# [1, 3, 224, 224]
#
# Output:
#
# [1, 2]
#
# Dynamic Axis 사용 X
# ============================================================
print(
    "\n[STEP 5] "
    "Export PyTorch -> ONNX"
)


torch.onnx.export(
    model,
    input_tensor,
    ONNX_FILE,
    export_params=True,
    opset_version=13,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["logits"],
    dynamic_axes=None,
    dynamo=False
)




##########################################################################




##########################################################################


print(
    "[OK] ONNX Exported :",
    ONNX_FILE
)


# ============================================================
# 22. ONNX 파일 존재 여부
# ============================================================
if not ONNX_FILE.exists():

    raise FileNotFoundError(
        f"ONNX export failed: "
        f"{ONNX_FILE}"
    )


# ============================================================
# 23. ONNX Model Load
# ============================================================
print(
    "\n[STEP 6] "
    "ONNX model validation"
)

onnx_model = onnx.load(ONNX_FILE)


##########################################################################




##########################################################################


# ============================================================
# 24. ONNX Checker
#
# ONNX Graph 구조 자체가 유효한지 검사
# ============================================================
##########################################################################




##########################################################################


print(
    "[OK] ONNX Checker PASS"
)


# ============================================================
# 25. ONNX Graph 기본 정보
# ============================================================
##########################################################################

onnx.checker.check_model(onnx_model)



##########################################################################


graph = onnx_model.graph


print(
    "[INFO] ONNX Graph Name :",
    graph.name
)


print(
    "[INFO] Number of Nodes :",
    len(graph.node)
)


print(
    "[INFO] Number of Inputs:",
    len(graph.input)
)


print(
    "[INFO] Number of Outputs:",
    len(graph.output)
)


# ============================================================
# 26. ONNX Input / Output 이름 확인
# ============================================================
print(
    "\n[INFO] ONNX Inputs"
)


for input_info in graph.input:

    print( " -",  input_info.name)


print("\n[INFO] ONNX Outputs")


for output_info in graph.output:

    print(" -", output_info.name)


# ============================================================
# 27. ONNX Runtime Session 생성
#
# CPUExecutionProvider 사용
#
# 여기서는 성능 비교가 목적이 아니다.
# 변환 결과 검증 목적이다.
# ============================================================
print(
    "\n[STEP 7] "
    "ONNX Runtime inference"
)

session = ort.InferenceSession(
    str(ONNX_FILE),
    providers=["CPUExecutionProvider"],
)


##########################################################################




##########################################################################


# ============================================================
# 28. ORT Input / Output 확인
# ============================================================
ort_input = (
    session.get_inputs()[0]
)


ort_output = (
    session.get_outputs()[0]
)


print(
    "[INFO] ORT Input Name :",
    ort_input.name
)


print(
    "[INFO] ORT Input Shape:",
    ort_input.shape
)


print(
    "[INFO] ORT Input Type :",
    ort_input.type
)


print(
    "[INFO] ORT Output Name:",
    ort_output.name
)


print(
    "[INFO] ORT Output Shape:",
    ort_output.shape
)


# ============================================================
# 29. PyTorch에서 사용한 Tensor와
#     완전히 동일한 값을 ONNX Runtime에 입력
#
# 중요한 비교 조건:
#
# 동일 이미지
# 동일 전처리
# 동일 Tensor 값
# 동일 Batch Size
#
# 달라지는 것은 Runtime뿐이다.
# ============================================================
onnx_input_np = (
    input_tensor
    .cpu()
    .numpy()
    .astype(
        np.float32
    )
)

onnx_output = session.run(
    [ort_output.name],
    {
        ort_input.name: onnx_input_np
    }
)[0]




##########################################################################




##########################################################################


print(
    "[INFO] ONNX Runtime Output Shape :",
    onnx_output.shape
)


print("[INFO] ONNX Runtime Logits :")
print(onnx_output)


# ============================================================
# 30. ONNX Runtime Top-1
# ============================================================
onnx_top1_index = int(
    np.argmax(onnx_output, axis=1)[0]
)


onnx_top1_class = class_names[onnx_top1_index]


print(
    "[INFO] ONNX Top-1 :",
    onnx_top1_class
)


# ============================================================
# 31. Stable Softmax 함수
#
# PyTorch / ONNX 두 결과를
# 확률 형태로도 비교한다.
#
# 직접 구현하는 이유:
#
# 두 Runtime의 Logits에
# 정확히 동일한 후처리를 적용하기 위함
# ============================================================
def softmax(logits):

    shifted = (
        logits - np.max(logits, axis=1, keepdims=True)
    )

    exp_values = np.exp(shifted)

    probabilities = (
        exp_values
        /
        np.sum(
            exp_values,
            axis=1,
            keepdims=True
        )
    )

    return probabilities


# ============================================================
# 32. PyTorch / ONNX Probability
# ============================================================
pytorch_probabilities = softmax(pytorch_output_np)


onnx_probabilities = softmax(onnx_output)


print("\n[INFO] PyTorch Probabilities")

print(pytorch_probabilities)


print("\n[INFO] ONNX Probabilities")

print(onnx_probabilities)


# ============================================================
# 33. Cat / Dog 확률
# ============================================================
pytorch_cat_probability = float(pytorch_probabilities[0, 0])
pytorch_dog_probability = float(pytorch_probabilities[0, 1])
onnx_cat_probability = float(onnx_probabilities[0, 0])
onnx_dog_probability = float(onnx_probabilities[0, 1])


print("\n[PyTorch]")

print(
    f"Cat : "
    f"{pytorch_cat_probability:.6f}"
)

print(
    f"Dog : "
    f"{pytorch_dog_probability:.6f}"
)


print(
    "\n[ONNX Runtime]"
)

print(
    f"Cat : "
    f"{onnx_cat_probability:.6f}"
)

print(
    f"Dog : "
    f"{onnx_dog_probability:.6f}"
)


# ============================================================
# 34. Absolute Error
#
# 각 Logit의 절대 오차 계산
# ============================================================
absolute_error = np.abs(pytorch_output_np - onnx_output)


max_abs_error = float(
    np.max(
        absolute_error
    )
)


mean_abs_error = float(
    np.mean(
        absolute_error
    )
)


print(
    "\n[STEP 8] "
    "PyTorch vs ONNX comparison"
)


print(
    "[INFO] Max Absolute Error  :",
    max_abs_error
)


print(
    "[INFO] Mean Absolute Error :",
    mean_abs_error
)


# ============================================================
# 35. np.allclose
#
# 두 결과가 설정한 허용 오차 범위 안에서
# 동일하다고 볼 수 있는지 확인
#
# |A-B| <= ATOL + RTOL * |B|
# ============================================================
allclose = bool(
    np.allclose(
        pytorch_output_np,
        onnx_output,
        rtol=RTOL,
        atol=ATOL,
    )
)


print(
    "[INFO] np.allclose :",
    allclose
)


# ============================================================
# 36. Top-1 Match
#
# 실제 최종 Prediction Class가 같은지도 확인
# ============================================================
top1_match = bool(
    pytorch_top1_index
    ==
    onnx_top1_index
)


print(
    "[INFO] Top-1 Match :",
    top1_match
)


# ============================================================
# 37. 최종 Conversion PASS
#
# 단순 Top-1만 같다고 성공으로 보지 않는다.
#
# 1. Logit 수치가 충분히 가까운가?
# 2. 최종 Top-1 결과도 같은가?
#
# 두 조건을 모두 확인한다.
# ============================================================
conversion_pass = bool(
    allclose
    and
    top1_match
)


print(
    "[INFO] Conversion PASS :",
    conversion_pass
)


# ============================================================
# 38. 결과 CSV 생성
# ============================================================
result = {

    "dataset_index":
        sample_dataset_index,

    "true_index":
        true_index,

    "true_class":
        true_class,


    "pytorch_cat_logit":
        float(
            pytorch_output_np[
                0,
                0
            ]
        ),

    "pytorch_dog_logit":
        float(
            pytorch_output_np[
                0,
                1
            ]
        ),


    "onnx_cat_logit":
        float(
            onnx_output[
                0,
                0
            ]
        ),

    "onnx_dog_logit":
        float(
            onnx_output[
                0,
                1
            ]
        ),


    "pytorch_cat_probability":
        pytorch_cat_probability,

    "pytorch_dog_probability":
        pytorch_dog_probability,


    "onnx_cat_probability":
        onnx_cat_probability,

    "onnx_dog_probability":
        onnx_dog_probability,


    "pytorch_top1_index":
        pytorch_top1_index,

    "pytorch_top1_class":
        pytorch_top1_class,


    "onnx_top1_index":
        onnx_top1_index,

    "onnx_top1_class":
        onnx_top1_class,


    "max_abs_error":
        max_abs_error,

    "mean_abs_error":
        mean_abs_error,


    "rtol":
        RTOL,

    "atol":
        ATOL,


    "allclose":
        allclose,

    "top1_match":
        top1_match,

    "conversion_pass":
        conversion_pass,
}


result_df = pd.DataFrame(
    [
        result
    ]
)


result_df.to_csv(
    RESULT_CSV_FILE,
    index=False,
)


# ============================================================
# 39. 최종 결과 출력
# ============================================================
print( "\n"  + "=" * 70)


print("A03 RESULT")


print("=" * 70)


print(
    "True Class       :",
    true_class
)


print(
    "PyTorch Top-1    :",
    pytorch_top1_class
)


print(
    "ONNX Top-1       :",
    onnx_top1_class
)


print(
    "Max Abs Error    :",
    max_abs_error
)


print(
    "Mean Abs Error   :",
    mean_abs_error
)


print(
    "np.allclose      :",
    allclose
)


print(
    "Top-1 Match      :",
    top1_match
)


print(
    "Conversion PASS  :",
    conversion_pass
)


print("\nONNX File:")

print(ONNX_FILE)


print("\nValidation Image:")

print(SAMPLE_IMAGE_FILE)


print("\nValidation CSV:")

print(RESULT_CSV_FILE)


print("=" * 70)