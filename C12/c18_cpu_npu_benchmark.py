from pathlib import Path

import csv
import os
import time

import cv2
import numpy as np
import onnxruntime as ort

from hailo_platform import (
    HEF, # 모델 로드
    VDevice, # 하드웨어 장치 관리
    ConfigureParams, # 모델 실행 환경 설정
    HailoStreamInterface, # PCLe 통신 인터페이스 지정
    InputVStreamParams, # 입력 스트림
    OutputVStreamParams, # 출력 스트림
    InferVStreams, # 추론 실행, 결과 수신
    FormatType, # 데이터타입 지정
)


# ============================================================
# C18 설정
# ============================================================

WARMUP = 10
REPEATS = 100

CLASS_NAMES = [
    "Cat",
    "Dog",
]


# ============================================================
# 경로 설정
# ============================================================

WORK_DIR = (
    Path.home()
    / "hailo_models"
    / "c18"
)

IMAGE_PATH = (
    WORK_DIR
    / "input.jpg"
)

ONNX_PATH = (
    WORK_DIR
    / "resnet18_catdog.onnx"
)

HEF_PATH = (
    WORK_DIR
    / "resnet18_catdog.hef"
)

RAW_PATH = (
    WORK_DIR
    / "c18_latency_raw.csv"
)

SUMMARY_PATH = (
    WORK_DIR
    / "c18_benchmark_summary.csv"
)


# ============================================================
# 1. 파일 존재 확인
# ============================================================

for path in [
    IMAGE_PATH,
    ONNX_PATH,
    HEF_PATH,
]:
    if not path.exists():
        raise FileNotFoundError(
            f"파일 없음: {path}"
        )


print("=" * 72)
print("C18 CPU vs Hailo-8L Benchmark")
print("=" * 72)

print("WORK DIR :", WORK_DIR)
print("IMAGE    :", IMAGE_PATH)
print("ONNX     :", ONNX_PATH)
print("HEF      :", HEF_PATH)
print("CPU cores:", os.cpu_count())
print("WARMUP   :", WARMUP)
print("REPEATS  :", REPEATS)


# ============================================================
# 2. 실제 JPEG 읽기
# ============================================================

image_bgr = cv2.imread(
    str(IMAGE_PATH)
)

if image_bgr is None:
    raise RuntimeError(
        f"이미지를 읽지 못했습니다: {IMAGE_PATH}"
    )

image_rgb = cv2.cvtColor(
    image_bgr,
    cv2.COLOR_BGR2RGB
)

print()
print("[ORIGINAL IMAGE]")
print("shape:", image_bgr.shape)
print("dtype:", image_bgr.dtype)


# ============================================================
# 3. HEF 입력/출력 정보
# ============================================================

hef = HEF(
    str(HEF_PATH)
)

input_info = (
    hef.get_input_vstream_infos()[0]
)

output_info = (
    hef.get_output_vstream_infos()[0]
)

input_height, input_width, input_channels = (
    input_info.shape
)

print()
print("[HEF]")
print("input name :", input_info.name)
print("input shape:", input_info.shape)
print("output name:", output_info.name)
print("output shape:", output_info.shape)


# ============================================================
# 4. 공통 Resize
# ============================================================

resized_rgb = cv2.resize(
    image_rgb,
    (
        input_width,
        input_height,
    )
)


# ============================================================
# 5. CPU용 입력 생성
# ============================================================

cpu_input = (
    resized_rgb.astype(
        np.float32
    )
    / 255.0
)

mean = np.array(
    [
        0.485,
        0.456,
        0.406,
    ],
    dtype=np.float32,
)

std = np.array(
    [
        0.229,
        0.224,
        0.225,
    ],
    dtype=np.float32,
)

cpu_input = (
    cpu_input
    -
    mean
) / std

cpu_input = np.transpose(
    cpu_input,
    (
        2,
        0,
        1,
    )
)

cpu_input = np.expand_dims(
    cpu_input,
    axis=0,
).astype(
    np.float32
)

print()
print("[CPU INPUT]")
print("shape:", cpu_input.shape)
print("dtype:", cpu_input.dtype)


# ============================================================
# 6. NPU용 입력 생성
# ============================================================

npu_input = np.expand_dims(
    resized_rgb,
    axis=0, # 배치형태로 하나 넣었음
).astype(
    np.uint8
)

print()
print("[NPU INPUT]")
print("shape:", npu_input.shape)
print("dtype:", npu_input.dtype)


# ============================================================
# 7. 통계 함수
# ============================================================

def make_stats(latencies_ms):

    values = np.asarray(
        latencies_ms,
        dtype=np.float64,
    )

    mean_ms = float(
        np.mean(values)
    )

    median_ms = float(
        np.median(values)
    )

    p95_ms = float(
        np.percentile(
            values,
            95
        )
    )

    min_ms = float(
        np.min(values)
    )

    max_ms = float(
        np.max(values)
    )

    std_ms = float(
        np.std(values)
    )

    throughput_fps = (
        1000.0
        /
        mean_ms
    )

    return {
        "mean_ms": mean_ms,
        "median_ms": median_ms,
        "p95_ms": p95_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "std_ms": std_ms,
        "throughput_fps": throughput_fps,
    }


# ============================================================
# 8. CPU ONNX Runtime Session
# ============================================================

print()
print("=" * 72)
print("[CPU] ONNX Runtime")
print("=" * 72)

session = ort.InferenceSession(
    str(ONNX_PATH),
    providers=[
        "CPUExecutionProvider"
    ],
)

onnx_input = (
    session.get_inputs()[0]
)

onnx_output = (
    session.get_outputs()[0]
)

cpu_input_name = (
    onnx_input.name
)

print(
    "provider:",
    session.get_providers()
)

print(
    "input:",
    onnx_input.name,
    onnx_input.shape,
    onnx_input.type,
)

print(
    "output:",
    onnx_output.name,
    onnx_output.shape,
    onnx_output.type,
)


# ============================================================
# 9. CPU 입력 Shape 검증
# ============================================================

expected_cpu_shape = (
    1,
    3,
    input_height,
    input_width,
)

if (
    tuple(cpu_input.shape)
    !=
    expected_cpu_shape
):
    raise RuntimeError(
        "CPU 입력 Shape가 예상과 다릅니다. "
        f"actual={cpu_input.shape}, "
        f"expected={expected_cpu_shape}"
    )


# ============================================================
# 10. CPU Warm-up
# ============================================================

print()
print(
    f"[CPU WARM-UP] {WARMUP} times"
)

for _ in range(
    WARMUP
):
    session.run(
        None,
        {
            cpu_input_name:
                cpu_input
        },
    )


# ============================================================
# 11. CPU Benchmark
# ============================================================

print(
    f"[CPU BENCHMARK] {REPEATS} times"
)

cpu_latencies = []
cpu_output = None

cpu_total_start = (
    time.perf_counter()
)

for _ in range(
    REPEATS
):
    t0 = time.perf_counter()

    result = session.run(
        None,
        {
            cpu_input_name:
                cpu_input
        },
    )

    t1 = time.perf_counter()

    cpu_latencies.append(
        (
            t1 - t0
        )
        * 1000.0
    )

    cpu_output = (
        result[0]
    )

cpu_total_s = (
    time.perf_counter()
    -
    cpu_total_start
)

cpu_stats = make_stats(
    cpu_latencies
)

cpu_total_throughput = (
    REPEATS
    /
    cpu_total_s
)

cpu_scores = np.asarray(
    cpu_output
).reshape(
    -1
)

if (
    len(cpu_scores)
    !=
    len(CLASS_NAMES)
):
    raise RuntimeError(
        "CPU Output class 수가 예상과 다릅니다. "
        f"actual={len(cpu_scores)}, "
        f"expected={len(CLASS_NAMES)}"
    )

cpu_top1 = int(
    np.argmax(
        cpu_scores
    )
)


print()
print("[CPU RESULT]")

print(
    "Top-1:",
    CLASS_NAMES[
        cpu_top1
    ]
)

print(
    "Mean   :",
    f"{cpu_stats['mean_ms']:.4f} ms"
)

print(
    "Median :",
    f"{cpu_stats['median_ms']:.4f} ms"
)

print(
    "P95    :",
    f"{cpu_stats['p95_ms']:.4f} ms"
)

print(
    "Min    :",
    f"{cpu_stats['min_ms']:.4f} ms"
)

print(
    "Max    :",
    f"{cpu_stats['max_ms']:.4f} ms"
)

print(
    "Std    :",
    f"{cpu_stats['std_ms']:.4f} ms"
)

print(
    "FPS(mean latency):",
    f"{cpu_stats['throughput_fps']:.2f}"
)

print(
    "FPS(total time):",
    f"{cpu_total_throughput:.2f}"
)


# ============================================================
# 12. Hailo-8L NPU
# ============================================================

print()
print("=" * 72)
print("[NPU] Hailo-8L")
print("=" * 72)

npu_latencies = []
npu_output = None

with VDevice() as target:

    configure_params = (
        ConfigureParams.create_from_hef(
            hef=hef,
            interface=(
                HailoStreamInterface.PCIe
            ),
        )
    )

    network_groups = (
        target.configure(
            hef,
            configure_params,
        )
    )

    network_group = (
        network_groups[0]
    )

    network_group_params = (
        network_group.create_params()
    )

    input_vstreams_params = (
        InputVStreamParams.make(
            network_group,
            format_type=(
                FormatType.UINT8
            ),
        )
    )

    output_vstreams_params = (
        OutputVStreamParams.make(
            network_group,
            format_type=(
                FormatType.FLOAT32
            ),
        )
    )

    input_data = {
        input_info.name:
            npu_input
    }

    with InferVStreams(
        network_group,
        input_vstreams_params,
        output_vstreams_params,
    ) as infer_pipeline:

        with network_group.activate(
            network_group_params
        ):

            print(
                f"[NPU WARM-UP] "
                f"{WARMUP} times"
            )

            for _ in range(
                WARMUP
            ):
                infer_pipeline.infer(
                    input_data
                )

            print(
                f"[NPU BENCHMARK] "
                f"{REPEATS} times"
            )

            npu_total_start = (
                time.perf_counter()
            )

            for _ in range(
                REPEATS
            ):
                t0 = (
                    time.perf_counter()
                )

                result = (
                    infer_pipeline.infer(
                        input_data
                    )
                )

                t1 = (
                    time.perf_counter()
                )

                npu_latencies.append(
                    (
                        t1 - t0
                    )
                    * 1000.0
                )

                npu_output = (
                    result[
                        output_info.name
                    ]
                )

            npu_total_s = (
                time.perf_counter()
                -
                npu_total_start
            )


# ============================================================
# 13. NPU 결과
# ============================================================

npu_stats = make_stats(
    npu_latencies
)

npu_total_throughput = (
    REPEATS
    /
    npu_total_s
)

npu_scores = np.asarray(
    npu_output
).reshape(
    -1
)

if (
    len(npu_scores)
    !=
    len(CLASS_NAMES)
):
    raise RuntimeError(
        "NPU Output class 수가 예상과 다릅니다. "
        f"actual={len(npu_scores)}, "
        f"expected={len(CLASS_NAMES)}"
    )

npu_top1 = int(
    np.argmax(
        npu_scores
    )
)


print()
print("[NPU RESULT]")

print(
    "Top-1:",
    CLASS_NAMES[
        npu_top1
    ]
)

print(
    "Mean   :",
    f"{npu_stats['mean_ms']:.4f} ms"
)

print(
    "Median :",
    f"{npu_stats['median_ms']:.4f} ms"
)

print(
    "P95    :",
    f"{npu_stats['p95_ms']:.4f} ms"
)

print(
    "Min    :",
    f"{npu_stats['min_ms']:.4f} ms"
)

print(
    "Max    :",
    f"{npu_stats['max_ms']:.4f} ms"
)

print(
    "Std    :",
    f"{npu_stats['std_ms']:.4f} ms"
)

print(
    "FPS(mean latency):",
    f"{npu_stats['throughput_fps']:.2f}"
)

print(
    "FPS(total time):",
    f"{npu_total_throughput:.2f}"
)


# ============================================================
# 14. CPU vs NPU 비교
# ============================================================

speedup = (
    cpu_stats[
        "mean_ms"
    ]
    /
    npu_stats[
        "mean_ms"
    ]
)

top1_match = (
    cpu_top1
    ==
    npu_top1
)


print()
print("=" * 72)
print("CPU vs NPU SUMMARY")
print("=" * 72)

print(
    "CPU Top-1 :",
    CLASS_NAMES[
        cpu_top1
    ]
)

print(
    "NPU Top-1 :",
    CLASS_NAMES[
        npu_top1
    ]
)

print(
    "Top-1 Match:",
    top1_match
)

print()

print(
    "CPU Mean Latency:",
    f"{cpu_stats['mean_ms']:.4f} ms"
)

print(
    "NPU Mean Latency:",
    f"{npu_stats['mean_ms']:.4f} ms"
)

print(
    "Speedup:",
    f"{speedup:.2f}x"
)

print()

print(
    "CPU Throughput:",
    f"{cpu_total_throughput:.2f} FPS"
)

print(
    "NPU Throughput:",
    f"{npu_total_throughput:.2f} FPS"
)


# ============================================================
# 15. Raw Latency CSV
# ============================================================

with open(
    RAW_PATH,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.writer(
        f
    )

    writer.writerow(
        [
            "iteration",
            "cpu_latency_ms",
            "npu_latency_ms",
        ]
    )

    for index in range(
        REPEATS
    ):
        writer.writerow(
            [
                index + 1,
                f"{cpu_latencies[index]:.8f}",
                f"{npu_latencies[index]:.8f}",
            ]
        )


# ============================================================
# 16. Summary CSV
# ============================================================

fieldnames = [
    "device",
    "model",
    "precision",
    "warmup",
    "repeats",
    "mean_ms",
    "median_ms",
    "p95_ms",
    "min_ms",
    "max_ms",
    "std_ms",
    "throughput_fps",
    "top1",
]

with open(
    SUMMARY_PATH,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    writer.writerow(
        {
            "device": "CPU",
            "model": "ResNet18 CatDog",
            "precision": "FP32",
            "warmup": WARMUP,
            "repeats": REPEATS,
            "mean_ms":
                f"{cpu_stats['mean_ms']:.8f}",
            "median_ms":
                f"{cpu_stats['median_ms']:.8f}",
            "p95_ms":
                f"{cpu_stats['p95_ms']:.8f}",
            "min_ms":
                f"{cpu_stats['min_ms']:.8f}",
            "max_ms":
                f"{cpu_stats['max_ms']:.8f}",
            "std_ms":
                f"{cpu_stats['std_ms']:.8f}",
            "throughput_fps":
                f"{cpu_total_throughput:.8f}",
            "top1":
                CLASS_NAMES[cpu_top1],
        }
    )

    writer.writerow(
        {
            "device": "Hailo-8L",
            "model": "ResNet18 CatDog",
            "precision": "INT8",
            "warmup": WARMUP,
            "repeats": REPEATS,
            "mean_ms":
                f"{npu_stats['mean_ms']:.8f}",
            "median_ms":
                f"{npu_stats['median_ms']:.8f}",
            "p95_ms":
                f"{npu_stats['p95_ms']:.8f}",
            "min_ms":
                f"{npu_stats['min_ms']:.8f}",
            "max_ms":
                f"{npu_stats['max_ms']:.8f}",
            "std_ms":
                f"{npu_stats['std_ms']:.8f}",
            "throughput_fps":
                f"{npu_total_throughput:.8f}",
            "top1":
                CLASS_NAMES[npu_top1],
        }
    )


# ============================================================
# 17. 완료
# ============================================================

print()
print("[SAVED]")
print(RAW_PATH)
print(SUMMARY_PATH)