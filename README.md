# NPU 기반 온디바이스 AI 모델 경량화·추론 실무

2026 수도권 ICT 이노베이션스퀘어 확산사업 · 인공지능(NPU) 과정 실습 저장소

## 과정 개요

| 항목 | 내용 |
|---|---|
| 과정명 | NPU 기반 온디바이스 AI 모델 경량화ㆍ추론 실무 |
| 교육시간 | 40시간 (주말 전일, 09:00~18:00) |
| 교육일 | 10/25, 11/1, 11/7, 11/8, 11/14 (10/31 휴강) |
| 방식 | 오프라인 |
| 실습 장비 | Raspberry Pi 5 (8GB), Raspberry Pi AI HAT+ (13 TOPS), Camera Module 3 |

## 커리큘럼

### 1주차: 온디바이스 추론 기초

| 차시 | 주제 | 내용 | 방식 |
|---|---|---|---|
| 1-1 | 온디바이스 AI와 Raspberry Pi NPU 이해 | Cloud AI vs On-device AI, CPU·GPU·NPU 역할 차이, NPU 연산 특성, 메모리·전력·연산 제약, Raspberry Pi 5와 NPU 가속기 구성 | 이론 |
| 1-2 | Raspberry Pi NPU 실행환경 구성 및 점검 | Raspberry Pi OS, AI HAT+ 연결 구조, NPU Driver·Runtime, 장치 인식 확인, Python 환경, Camera 연결 및 입출력 점검 | 이론/실습 |
| 1-3 | AI 추론 파이프라인 | 사전학습 모델 활용, 이미지·카메라 입력, Resize·Normalize, Tensor 변환, 추론·후처리, CPU/NPU 기본 실행 및 Baseline 측정 | 이론/실습 |
| 1-4 | ONNX 변환 및 검증 | ONNX 개념, ONNX Export, Input/Output Shape, Opset, ONNX Runtime 실행, 원본/변환 모델 결과 및 Latency 비교 | 이론/실습 |
| 1-5 | 모델 입출력 구조와 배포 오류 분석 | NCHW·NHWC, RGB·BGR, Shape·Data Type, Resize·Normalize 불일치, 전처리·입출력 오류 분석 및 해결 | 이론/실습 |

### 2주차: 경량화와 NPU 배포

| 차시 | 주제 | 내용 | 방식 |
|---|---|---|---|
| 2-1 | FP32·FP16·INT8 모델 경량화 | 표현 방식, 모델 크기와 연산량, FP16 변환, INT8 Quantization, Calibration, 경량화 전·후 비교 | 이론 |
| 2-2 | NPU용 모델 변환 및 배포 | ONNX와 NPU 실행모델 관계, NPU 지원 연산, 최적화·Quantization·Compile 구조, NPU 실행모델 생성 및 배포 | 이론 |
| 2-3 | CPU·NPU 성능 및 경량화 효과 비교 | 동일 조건 CPU/NPU Benchmark, FP32/FP16/INT8 비교, Latency·FPS·Memory·정확도 분석 | 이론/실습 |
| 2-4 | 온디바이스 추론 파이프라인 최적화 | 입력 Resolution, Camera FPS, Frame Sampling, 전·후처리 비용, CPU 부하, End-to-End Latency 최적화 | 이론/실습 |
| 2-5 | 온디바이스 AI 응용 프로젝트 | 변환·경량화·NPU 추론을 하나의 Pipeline으로 통합, 이미지/Camera 응용 구현, 최적화 전·후 비교 및 발표 | 실습 |

## 폴더 구조

```
npu/
├── day1/            # 1일차 실습
│   └── test.py      # 환경 확인 (numpy, torch 버전 출력)
├── requirements.txt
├── 명령어_복습.md    # 터미널·venv·pip 명령어 정리
└── README.md
```

## 개발 환경 (로컬 PC, macOS)

- Python 3.12.10
- PyTorch 2.12.0 (CPU only)
- PyCharm, 인터프리터: 프로젝트의 `npu_env`

```bash
python -m venv npu_env
source npu_env/bin/activate          # Windows: npu_env\Scripts\activate
pip install -r requirements.txt
pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu
```

환경 확인:

```bash
python day1/test.py
```
