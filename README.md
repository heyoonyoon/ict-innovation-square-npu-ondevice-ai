<div align="center">

# NPU 기반 온디바이스 AI

**2026 수도권 ICT 이노베이션스퀘어 인공지능(NPU) 과정 실습 저장소**

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.12-EE4C2C?logo=pytorch&logoColor=white)
![ONNX](https://img.shields.io/badge/ONNX-Runtime-005CED?logo=onnx&logoColor=white)
![Hailo](https://img.shields.io/badge/Hailo--8L-13_TOPS-1f2937)
![Raspberry Pi](https://img.shields.io/badge/Raspberry_Pi_5-AI_HAT+-A22846?logo=raspberrypi&logoColor=white)

<br>

<img src="assets/diagrams/01_pipeline.png" alt="학습, ONNX 변환, 양자화, 컴파일, 온디바이스 추론 5단계 파이프라인" width="900">

</div>

<br>

## 프로젝트 1. 드론 경보 분류

카메라 화면 전체를 보고 드론, 새, 빈 하늘 가운데 하나로 판별해 드론일 때 경보를 띄웁니다.

<div align="center">
  <img src="assets/p1_alarm_demo.gif" alt="새 사진에서는 경보가 없고 드론 사진에서 DRONE ALERT가 뜨는 시연" width="560">
  <br>
  <img src="assets/p1_setup.jpg" alt="Pi 카메라 모듈로 노트북 화면의 드론 사진을 비추는 모습" width="560">
</div>

| 항목 | 내용 |
|---|---|
| 모델 | ResNet18 전이학습, 3클래스 |
| 데이터 | drone vs bird 합성 사진 6,000장 (CC BY 4.0) |
| 검증 정확도 | 96.33% |
| NPU 결과 | 카메라로 찍은 드론 사진 drone 0.966 (PC ONNX 0.969) |
| 코드 | [`DRONE_LAB/`](DRONE_LAB) |

<br>

## 프로젝트 2. 드론 탐지와 추적

드론에 박스를 치고 번호를 붙여 움직임을 따라가며, 같은 드론이 일정 시간 이상 잡히면 경보를 띄웁니다.

<div align="center">
  <img src="assets/p2_track_demo.gif" alt="실제 드론 영상을 Pi 카메라로 비춰 박스가 드론을 따라가는 시연" width="560">
  <br>
  <img src="assets/drone_track_demo.png" alt="합성 테스트 영상에서 드론과 새에 번호 박스와 이동 경로가 그려진 추적 결과" width="560">
</div>

| 항목 | 내용 |
|---|---|
| 모델 | YOLOv8n 전이학습, 4클래스 (drone, bird, airplane, helicopter) |
| 학습 | Colab T4 GPU, mAP50 0.845 |
| NPU 속도 | 단독 104 FPS, 카메라부터 화면까지 약 45 FPS |
| 컴파일 | AWS EC2(x86 Ubuntu)에서 Hailo DFC 3.33 |
| 코드 | [`DRONE_TRACK/`](DRONE_TRACK) |

<br>

## 배운 것

<div align="center">
  <img src="assets/diagrams/02_transfer_learning.png" alt="사전학습 모델의 CNN은 고정하고 마지막 FC만 새로 학습하는 전이학습" width="900">
  <img src="assets/diagrams/03_pytorch_loop.png" alt="데이터, 순전파, 손실, 역전파, 가중치 갱신을 epoch마다 반복하는 PyTorch 학습 루프" width="900">
  <img src="assets/diagrams/08_tensor_format.png" alt="PyTorch는 NCHW float32, Hailo는 NHWC uint8 입력" width="900">
  <img src="assets/diagrams/04_quantization.png" alt="Calibration과 scale로 FP32를 INT8로 바꾸고 모델 크기를 4배 줄이는 양자화" width="900">
  <img src="assets/diagrams/05_hailo_compiler.png" alt="Docker 안의 Hailo 컴파일러가 Parse, Optimize, Compile로 HEF를 만드는 과정" width="900">
  <img src="assets/diagrams/06_pi_inference.png" alt="카메라, 전처리, NPU 추론, 후처리, 추적과 경보로 이어지는 라즈베리파이 실시간 추론" width="900">
  <img src="assets/diagrams/07_tracker.png" alt="후보, 포착, 추적, 놓침, 해제로 이어지는 추적기 상태" width="900">
</div>

<br>

## 실습 현장

<div align="center">
  <img src="assets/course_banner.jpg" alt="2026년 10월 25일부터 11월 14일까지 진행된 교육 과정 안내판" width="240">
  <img src="assets/badge.jpg" alt="NPU 기반 온디바이스AI 모델 경량화 추론 실무 교육 교육생 명찰" width="240">
  <br>
  <img src="assets/hef_compile_vm.jpg" alt="노트북 VM의 Hailo Docker에서 HEF를 컴파일하고 옆에 라즈베리파이를 연결한 모습" width="490">
</div>

| 장비 | 역할 |
|---|---|
| Raspberry Pi 5 8GB | 추론 본체 |
| AI HAT+ (Hailo-8L, 13 TOPS) | NPU |
| Camera Module 3 | 영상 입력 |

| 폴더 | 내용 |
|---|---|
| `C03` ~ `ch09` | 차시별 실습 (추론, Latency, ONNX 변환, 양자화, HEF) |
| `NPU_LAB/ADD` | 고양이와 강아지 전이학습 |
| `DRONE_LAB` | 프로젝트 1 |
| `DRONE_TRACK` | 프로젝트 2 |
