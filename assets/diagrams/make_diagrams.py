# README 도식 생성 스크립트 (토스 스타일: 회색 바탕, 흰 카드, 파란 강조)
# 실행 : python assets/diagrams/make_diagrams.py  → assets/diagrams/*.png

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

OUT = Path(__file__).resolve().parent
FONT_DIR = Path.home() / "Library" / "Fonts"
F = {w: fm.FontProperties(fname=str(FONT_DIR / f"Pretendard-{w}.otf"))
     for w in ["Regular", "Medium", "SemiBold", "Bold", "ExtraBold"]}

# 색
BG, CARD = "#F2F4F6", "#FFFFFF"
INK, SUB, MUTE, LINE = "#191F28", "#4E5968", "#8B95A1", "#D1D6DB"
BLUE, BLUE_T, BLUE_D = "#3182F6", "#E8F3FF", "#1B64DA"
GREEN, GREEN_T = "#03B26C", "#E5F8EF"
RED, RED_T = "#F04452", "#FFEEEE"
GRAY_T = "#F2F4F6"


# ------------------------------------------------------------
# 그리기 도우미 (좌표는 픽셀, 위에서 아래로)
# ------------------------------------------------------------
class Canvas:
    def __init__(self, W, H, title, sub=None):
        self.W, self.H = W, H
        self.fig = plt.figure(figsize=(W / 100, H / 100), dpi=200)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, W)
        self.ax.set_ylim(H, 0)
        self.ax.axis("off")
        self.ax.add_patch(FancyBboxPatch((0, 0), W, H, boxstyle="round,pad=0,rounding_size=36",
                                         fc=BG, ec="none"))
        self.text(56, 64, title, 25, "ExtraBold", INK, va="center")
        if sub:
            self.text(56, 108, sub, 14.5, "Medium", MUTE, va="center")

    def text(self, x, y, s, size, weight="Regular", color=INK, ha="left", va="center"):
        self.ax.text(x, y, s, fontproperties=F[weight], fontsize=size, color=color, ha=ha, va=va)

    def card(self, x, y, w, h, fc=CARD, r=26, ec="none", lw=0):
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                         fc=fc, ec=ec, lw=lw))

    def chip(self, x, y, s, fc=BLUE_T, tc=BLUE_D, size=12, ha="center"):
        self.ax.text(x, y, s, fontproperties=F["SemiBold"], fontsize=size, color=tc, ha=ha, va="center",
                     bbox=dict(boxstyle="round,pad=0.45,rounding_size=0.8", fc=fc, ec="none"))

    def num(self, x, y, n, fc=BLUE):
        self.ax.add_patch(Circle((x, y), 17, fc=fc, ec="none"))
        self.text(x, y + 1, str(n), 13, "Bold", "white", ha="center")

    def arrow(self, x1, y1, x2, y2, color=MUTE, lw=2.2, rad=0.0, style="-|>"):
        self.ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=16,
                                          lw=lw, color=color, connectionstyle=f"arc3,rad={rad}",
                                          shrinkA=0, shrinkB=0))

    def save(self, name):
        self.fig.savefig(OUT / name, transparent=True)
        plt.close(self.fig)
        print("saved", name)


def step_cards(c, steps, y, h, x0=56, gap=36, title_size=17):
    """번호 원 + 제목 + 설명 + 아래 칩. 가로로 나열하고 사이에 화살표."""
    n = len(steps)
    w = (c.W - 2 * x0 - gap * (n - 1)) / n
    for i, s in enumerate(steps):
        x = x0 + i * (w + gap)
        c.card(x, y, w, h)
        c.num(x + 40, y + 44, i + 1, s.get("color", BLUE))
        c.text(x + 68, y + 45, s["title"], title_size, "Bold", INK)
        for j, line in enumerate(s["lines"]):
            c.text(x + 28, y + 96 + j * 30, line, 13.5, "Regular", SUB)
        if s.get("chip"):
            c.chip(x + 28, y + h - 36, s["chip"], ha="left",
                   fc=s.get("chip_fc", BLUE_T), tc=s.get("chip_tc", BLUE_D))
        if i < n - 1:
            c.arrow(x + w + 6, y + h / 2, x + w + gap - 6, y + h / 2)
    return w


# ------------------------------------------------------------
# 1. 전체 파이프라인
# ------------------------------------------------------------
c = Canvas(1800, 470, "모델 학습부터 NPU 추론까지", "모델 파일은 단계마다 형식을 바꾸고, 마지막 HEF만 라즈베리파이로 옮겨 실행합니다")
step_cards(c, [
    dict(title="학습", lines=["사전학습 모델 불러오기", "내 데이터로 전이학습", "Mac, Colab GPU"], chip=".pt"),
    dict(title="ONNX 변환", lines=["PyTorch 그래프 내보내기", "출력 오차 1e-5 검증", "Mac"], chip=".onnx"),
    dict(title="양자화", lines=["calib 사진 100장", "FP32를 INT8로", "x86 Linux, Hailo DFC"], chip=".har"),
    dict(title="컴파일", lines=["Hailo-8L 전용", "실행 파일 생성", "x86 Linux, Hailo DFC"], chip=".hef"),
    dict(title="온디바이스 추론", lines=["카메라 실시간 입력", "NPU 추론, 화면과 경보", "Raspberry Pi 5"],
         chip="HailoRT", color=RED, chip_fc=RED_T, chip_tc=RED),
], y=150, h=270)
c.save("01_pipeline.png")


# ------------------------------------------------------------
# 2. 전이학습
# ------------------------------------------------------------
c = Canvas(1800, 600, "전이학습", "이미 수많은 사진으로 배운 모델에서 판단하는 마지막 층만 바꿔 다시 가르칩니다")

def model_row(c, x, y, label, head, head_sub, head_color, head_fc, lock_text, lock_fc, lock_tc, head_chip):
    c.card(x, y, 800, 360)
    c.text(x + 36, y + 46, label, 17, "Bold", INK)
    bx, by = x + 36, y + 100
    for k in range(4):                      # CNN 블록 4개
        c.card(bx + k * 104, by, 88, 150, fc=GRAY_T, r=16)
        c.text(bx + k * 104 + 44, by + 75, f"Conv\n{k + 1}", 12.5, "SemiBold", SUB, ha="center")
    c.text(bx, by + 186, "CNN 특징 추출  선, 모양, 질감", 13.5, "Medium", SUB)
    c.chip(bx + 6, by + 226, lock_text, fc=lock_fc, tc=lock_tc, ha="left")
    c.arrow(bx + 420, by + 75, bx + 470, by + 75)
    c.card(bx + 482, by, 246, 150, fc=head_fc, r=16)
    c.text(bx + 605, by + 58, head, 16, "Bold", head_color, ha="center")
    c.text(bx + 605, by + 98, head_sub, 12.5, "Medium", head_color, ha="center")
    c.text(bx + 482, by + 186, "FC 판단  무엇인지 결정", 13.5, "Medium", SUB)
    c.chip(bx + 488, by + 226, head_chip[0], fc=head_chip[1], tc=head_chip[2], ha="left")

model_row(c, 56, 160, "사전학습 모델  ImageNet 1,000종", "FC 1000", "고양이, 자동차 ...", SUB, GRAY_T,
          "그대로 가져옴", GRAY_T, SUB, ("기존 판단", GRAY_T, SUB))
c.arrow(870, 340, 930, 340, color=BLUE, lw=3)
model_row(c, 944, 160, "내 모델  드론 경보", "FC 3", "드론, 새, 하늘", BLUE_D, BLUE_T,
          "고정, 학습 안 함", GRAY_T, SUB, ("새로 학습", BLUE, "white"))
c.text(1700, 520 + 30, "적은 데이터와 짧은 시간으로 96% 정확도", 13.5, "SemiBold", BLUE_D, ha="right")
c.save("02_transfer_learning.png")


# ------------------------------------------------------------
# 3. PyTorch 학습 루프
# ------------------------------------------------------------
c = Canvas(1800, 560, "PyTorch 학습 루프", "한 묶음씩 예측하고, 틀린 만큼 가중치를 고치는 일을 epoch마다 반복합니다")
w = step_cards(c, [
    dict(title="데이터", lines=["DataLoader", "32장씩 묶음", "전처리와 증강"], chip="batch"),
    dict(title="순전파", lines=["output = model(x)", "확률 계산", ""], chip="forward"),
    dict(title="손실 계산", lines=["loss = criterion(", "    output, y)", "정답과의 차이"], chip="loss"),
    dict(title="역전파", lines=["loss.backward()", "층마다 기울기 계산", ""], chip="gradient"),
    dict(title="가중치 갱신", lines=["optimizer.step()", "기울기 반대로 조금 이동", ""], chip="update"),
], y=150, h=250)
c.arrow(1744 - w / 2, 412, 56 + w / 2, 412, color=BLUE, lw=2.4, rad=-0.12)
c.text(900, 468, "epoch 반복", 14, "Bold", BLUE_D, ha="center")
c.chip(900, 512, "검증 정확도가 가장 높을 때의 가중치를 best.pt로 저장", fc=CARD, tc=SUB, size=12.5)
c.save("03_pytorch_loop.png")


# ------------------------------------------------------------
# 4. 양자화
# ------------------------------------------------------------
c = Canvas(1800, 560, "양자화와 Calibration", "실수 가중치를 -127부터 127 사이 정수로 바꿔 모델을 4배 작게 만듭니다")
c.card(56, 150, 1000, 360)
c.text(92, 196, "INT8 변환 방법", 17, "Bold", INK)
rows = [("1", "Calibration", "실제 사진 100장을 넣어 층마다 값의 범위를 측정"),
        ("2", "scale", "scale = 가장 큰 절댓값 ÷ 127"),
        ("3", "반올림", "정수 = round(실수 ÷ scale)"),
        ("4", "자르기", "-127보다 작거나 127보다 크면 끝값으로 고정")]
for i, (n, k, v) in enumerate(rows):
    yy = 256 + i * 62
    c.num(110, yy, n)
    c.text(142, yy, k, 15, "Bold", INK)
    c.text(300, yy, v, 14, "Regular", SUB)

c.card(1092, 150, 652, 360)
c.text(1128, 196, "모델 크기  드론 ResNet18", 17, "Bold", INK)
bars = [("FP32", 42.6, LINE, SUB), ("FP16", 21.3, "#90C2FF", BLUE_D), ("INT8", 10.7, BLUE, BLUE_D)]
for i, (k, v, col, tc) in enumerate(bars):
    yy = 262 + i * 76
    c.text(1128, yy, k, 15, "Bold", INK)
    c.card(1210, yy - 20, 380 * v / 42.6, 40, fc=col, r=12)
    c.text(1222 + 380 * v / 42.6, yy, f"{v} MiB", 14, "SemiBold", tc)
c.text(1128, 478, "포화 비율 0%, 첫 Conv 최대 오차 0.004", 13, "Medium", MUTE)
c.save("04_quantization.png")


# ------------------------------------------------------------
# 5. Hailo 컴파일러
# ------------------------------------------------------------
c = Canvas(1800, 640, "Hailo 컴파일러로 HEF 만들기", "x86 리눅스에서만 동작하는 Hailo Dataflow Compiler로 ONNX를 NPU 실행 파일로 바꿉니다")
env = [("Mac", "ONNX와 calib 준비"), ("AWS EC2 또는 VM", "Ubuntu 22.04 x86_64"),
       ("Docker", "Hailo AI SW Suite 2025-10"), ("Raspberry Pi", "HailoRT 4.23")]
ew = (1800 - 112 - 3 * 36) / 4
for i, (k, v) in enumerate(env):
    x = 56 + i * (ew + 36)
    c.card(x, 150, ew, 96, fc=CARD if i != 2 else BLUE_T)
    c.text(x + 28, 184, k, 15.5, "Bold", INK if i != 2 else BLUE_D)
    c.text(x + 28, 218, v, 13, "Medium", SUB)
    if i < 3:
        c.arrow(x + ew + 6, 198, x + ew + 30, 198)
c.text(56, 290, "Docker 안에서 hailomz 명령 한 줄로 세 단계 실행", 14, "SemiBold", SUB)
step_cards(c, [
    dict(title="Parse", lines=["ONNX를 Hailo 내부 표현으로", "Model Zoo yolov8n 설정 재사용", "--ckpt 로 내 모델 지정"], chip="ONNX → HAR"),
    dict(title="Optimize", lines=["calib 100장으로 범위 측정", "INT8 양자화", "GPU 없으면 최적화 레벨 0"], chip="HAR → 양자화 HAR"),
    dict(title="Compile", lines=["칩 자원에 층 배치", "YOLOv8n은 3조각으로 분할", "NMS 후처리 포함"], chip="HAR → HEF"),
], y=320, h=270, title_size=18)
c.save("05_hailo_compiler.png")


# ------------------------------------------------------------
# 6. 라즈베리파이 실시간 추론
# ------------------------------------------------------------
c = Canvas(1800, 560, "라즈베리파이 실시간 추론", "카메라 한 프레임마다 아래 순서를 반복합니다")
step_cards(c, [
    dict(title="카메라", lines=["Camera Module 3", "1280×720", "Picamera2"], chip="BGR 프레임"),
    dict(title="전처리", lines=["비율 유지 640 맞춤", "남는 부분 회색 채움", "정규화는 HEF 안에서"], chip="UINT8 NHWC"),
    dict(title="NPU 추론", lines=["Hailo-8L 13 TOPS", "YOLOv8n HEF", ""], chip="11 ms", color=RED, chip_fc=RED_T, chip_tc=RED),
    dict(title="후처리", lines=["NMS로 중복 박스 정리", "점수 0.2 미만 제거", "Pi CPU에서 실행"], chip="박스 목록"),
    dict(title="추적과 경보", lines=["칼만 필터 위치 예측", "IoU로 같은 물체 연결", "드론이면 경보"], chip="ID와 경로"),
], y=150, h=270)
for i, (k, v) in enumerate([("NPU 단독", "104 FPS"), ("카메라부터 화면까지", "약 45 FPS"), ("칩 온도", "37°C")]):
    x = 56 + i * 300
    c.text(x, 476, k, 13.5, "Medium", MUTE)
    c.text(x, 512, v, 19, "ExtraBold", INK)
c.save("06_pi_inference.png")


# ------------------------------------------------------------
# 7. 추적기 상태
# ------------------------------------------------------------
c = Canvas(1800, 520, "추적기", "프레임마다 찾은 박스를 이전 물체와 짝지어 같은 번호를 이어 붙입니다")
states = [("후보", "처음 보임", CARD, SUB), ("포착", "3프레임 연속", BLUE_T, BLUE_D),
          ("추적", "번호와 이동 경로", BLUE, "white"), ("놓침", "예측 위치 유지", "#FFF4E5", "#E67700"),
          ("해제", "15프레임 못 찾음", CARD, MUTE)]
sw = (1800 - 112 - 4 * 60) / 5
for i, (k, v, fc, tc) in enumerate(states):
    x = 56 + i * (sw + 60)
    c.card(x, 170, sw, 150, fc=fc)
    c.text(x + sw / 2, 222, k, 20, "ExtraBold", tc, ha="center")
    c.text(x + sw / 2, 270, v, 13.5, "Medium", tc if fc != BLUE else "white", ha="center")
    if i < 4:
        c.arrow(x + sw + 10, 245, x + sw + 50, 245)
x_track, x_lost = 56 + 2 * (sw + 60) + sw / 2, 56 + 3 * (sw + 60) + sw / 2
c.arrow(x_lost, 330, x_track, 330, color="#E67700", rad=-0.35)
c.text((x_track + x_lost) / 2, 432, "다시 찾으면 같은 번호", 13, "SemiBold", "#E67700", ha="center")
c.chip(56, 462, "경보 조건  drone으로 포착된 물체가 10프레임 이상", fc=RED_T, tc=RED, ha="left", size=13)
c.save("07_tracker.png")


# ------------------------------------------------------------
# 8. 텐서 형식
# ------------------------------------------------------------
c = Canvas(1800, 470, "같은 사진, 다른 입력 형식", "배포 오류의 대부분은 모델마다 다른 입력 형식에서 생깁니다")
cols = [("PyTorch, ONNX", BLUE_T, BLUE_D, [("모양", "NCHW  1×3×224×224"), ("자료형", "float32"),
                                           ("값", "0~1로 나누고 평균, 표준편차 정규화"), ("색", "RGB")]),
        ("Hailo HEF", RED_T, RED, [("모양", "NHWC  1×224×224×3"), ("자료형", "uint8"),
                                   ("값", "0~255 그대로  정규화는 칩 안에서"), ("색", "RGB  OpenCV는 BGR이라 변환")])]
cw = (1800 - 112 - 36) / 2
for i, (title, fc, tc, items) in enumerate(cols):
    x = 56 + i * (cw + 36)
    c.card(x, 150, cw, 270)
    c.chip(x + 32, 192, title, fc=fc, tc=tc, ha="left", size=14)
    for j, (k, v) in enumerate(items):
        yy = 254 + j * 44
        c.text(x + 34, yy, k, 14, "Medium", MUTE)
        c.text(x + 150, yy, v, 15, "SemiBold", INK)
c.save("08_tensor_format.png")
