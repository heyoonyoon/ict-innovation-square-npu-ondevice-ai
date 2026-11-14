# ============================================================
# c03_make_video.py
#
# 하늘 배경 위로 드론·새가 날아다니는 "합성 테스트 영상"과
# 정답(Ground Truth) 박스 JSON 을 만든다.
#
# 쓰임새
#   1) 탐지 모델이 없어도 Mac 에서 트래커(추적기)를 먼저 시험해 보기
#   2) 나중에 노트북 전체화면으로 재생 → Pi 카메라로 찍어서 데모
#
# 결과물 (video/ 폴더)
#   c03_test.mp4      : 1280x720, 30fps, 20초 영상
#   c03_test_gt.json  : 프레임별 정답 박스
#   c03_preview.png   : 정답 박스를 그린 샘플 8장 (눈으로 확인용)
#
# 실행
#   ../npu_env/bin/python c03_make_video.py
#   (ffmpeg 필요: brew install ffmpeg)
#
# 원본 이미지 : DRONE_LAB/data/drone_cls/{drone,bird,sky}/*.jpg
#   (Hugging Face athkot/drone-vs-bird-synthetic-10k, CC BY 4.0)
# ============================================================
from pathlib import Path
import json
import random
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


# ============================================================
# 1. 경로 설정
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent

CLS_DIR = SCRIPT_DIR.parent / "DRONE_LAB" / "data" / "drone_cls"

OUT_DIR = SCRIPT_DIR / "video"
VIDEO_PATH = OUT_DIR / "c03_test.mp4"
GT_PATH = OUT_DIR / "c03_test_gt.json"
PREVIEW_PATH = OUT_DIR / "c03_preview.png"


# ============================================================
# 2. 설정
#
# 클래스 번호는 원본 데이터와 같게 맞춘다
#   0 = drone, 1 = bird, 2 = airplane, 3 = helicopter
# ============================================================
SEED = 42
W, H = 1280, 720
FPS = 30
SECONDS = 20
N_FRAMES = FPS * SECONDS

CLASSES = ["drone", "bird", "airplane", "helicopter"]

# 시나리오 : 물체 하나 = (id, 클래스, 시작초, 끝초, 경유점들, 시작크기, 끝크기)
#   경유점 : 화면 좌표 (x, y). 시작~끝 시간 동안 경유점을 차례로 직선 이동
#   크기   : 화면에 보이는 "물체" 크기(px). 붙이는 사진은 그 2배 크기
SCENARIO = [
    # (a) 0~4.5초 : 드론 1대가 천천히 왼쪽 → 오른쪽
    (1, 0, 0.0, 4.5, [(-120, 300), (1400, 260)], 120, 150),
    # (b) 4.5~8.5초 : 드론과 새가 동시에, 경로가 X 자로 교차
    (2, 0, 4.5, 8.5, [(-120, 180), (1400, 540)], 130, 130),
    #     (가운데서 살짝 겹쳤다가 지나감 → 트래커가 id 를 안 바꾸는지 확인)
    (3, 1, 4.5, 8.5, [(1400, 260), (-120, 600)], 110, 120),
    # (c) 8.5~10.5초 : 빠른 드론 (2초 만에 화면 가로지름)
    (4, 0, 8.5, 10.5, [(-120, 420), (1400, 360)], 100, 100),
    # (d) 10.5~15초 : 지그재그로 방향을 바꾸는 드론 (마지막엔 되돌아감)
    (5, 0, 10.5, 15.0,
     [(-120, 360), (300, 150), (550, 550), (850, 180), (1050, 500),
      (700, 380), (-140, 300)], 160, 200),
    # (e) 15~18초 : 새만 등장 (경보가 울리면 안 되는 구간)
    (6, 1, 15.0, 18.0, [(1400, 250), (-120, 330)], 140, 150),
    (7, 1, 15.5, 18.0, [(-120, 520), (1400, 450)], 90, 100),
    # (f) 18~20초 : 아무것도 없는 하늘
]


# ============================================================
# 3. 배경 만들기
#
# 하늘 사진 한 장(256x256)의 가운데 16:9 부분을 잘라
# 1280x720 으로 부드럽게 키운다. (블러로 JPEG 깍두기 제거)
# ============================================================
rng = random.Random(SEED)

def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)

def border_pixels(img, k=40):
    # 사진 테두리(물체가 없는 하늘 부분)의 픽셀들 (N, 3)
    return np.concatenate([img[:k].reshape(-1, 3), img[-k:].reshape(-1, 3),
                           img[:, :k].reshape(-1, 3), img[:, -k:].reshape(-1, 3)])

def border_mean(img):
    return border_pixels(img).mean(axis=0)

# "파란 하늘"(B 가 R 보다 40 이상 큼, 너무 밝거나 어둡지 않음) 중에서
# 얼룩이 가장 적은(표준편차가 작은) 사진을 고른다
sky_paths = sorted((CLS_DIR / "sky").glob("*.jpg"))
candidates = []
for p in rng.sample(sky_paths, 200):
    img = load_rgb(p)
    m = img.mean(axis=(0, 1))
    if m[2] - m[0] > 40 and 120 < m.mean() < 220 and m.max() < 245:
        candidates.append((img.std(axis=(0, 1)).mean(), p))
candidates.sort()
sky_path = candidates[0][1]

sky = Image.open(sky_path).convert("RGB")
sky = sky.crop((0, 56, 256, 200))                      # 256x144 = 16:9
sky = sky.resize((W, H), Image.BICUBIC).filter(ImageFilter.GaussianBlur(6))
BACKGROUND = np.asarray(sky, dtype=np.float32)
BG_MEAN = BACKGROUND.mean(axis=(0, 1))
print(f"배경 : {sky_path.name}  평균색 {BG_MEAN.round(1)}")


# ============================================================
# 4. 물체 사진 고르기
#
# 각 물체마다 "테두리가 깨끗한 하늘"인 사진을 고른다.
#   점수 = 테두리 평균색과 배경색의 차이 + 테두리 얼룩(표준편차) x 3
#   (테두리에 건물·구름·렌즈가 찍힌 사진은 얼룩 점수가 커서 빠진다)
# 그래도 남는 색 차이는 "평균색 차이만큼 더해서" 맞춘다.
# ============================================================
def pick_sprite(cls_name, used):
    paths = sorted((CLS_DIR / cls_name).glob("*.jpg"))
    best = None
    for p in rng.sample(paths, 1000):
        if p in used:
            continue
        img = load_rgb(p)
        ring = border_pixels(img)
        diff = np.abs(ring.mean(axis=0) - BG_MEAN).sum() + ring.std(axis=0).sum() * 3
        if best is None or diff < best[0]:
            best = (diff, p, img)
    used.add(best[1])
    img = best[2] + (BG_MEAN - border_mean(best[2]))   # 색 보정
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))

used = set()
SPRITES = {}
for obj_id, cls, *_ in SCENARIO:
    SPRITES[obj_id] = pick_sprite(CLASSES[cls], used)


# ============================================================
# 5. 부드러운 알파 마스크
#
# 사진을 네모 그대로 붙이면 경계가 보인다.
# 가운데(물체 부분)는 1, 바깥으로 갈수록 0 이 되는 원형 마스크를 쓴다.
# r = 중심에서의 거리 / 사진 크기  →  r<0.36 은 불투명, r>0.5 는 투명
# ============================================================
def make_mask(size):
    yy, xx = np.mgrid[0:size, 0:size]
    c = (size - 1) / 2
    r = np.sqrt((xx - c) ** 2 + (yy - c) ** 2) / size
    t = np.clip((0.5 - r) / (0.5 - 0.36), 0, 1)
    return (t * t * (3 - 2 * t))[..., None]           # smoothstep


# ============================================================
# 6. 시간 t 에서 물체의 위치·크기 계산
# ============================================================
def object_state(t, obj):
    obj_id, cls, t0, t1, points, s0, s1 = obj
    if not (t0 <= t < t1):
        return None
    u = (t - t0) / (t1 - t0)                         # 0 → 1 진행률
    seg = u * (len(points) - 1)                       # 몇 번째 구간인지
    i = min(int(seg), len(points) - 2)
    f = seg - i
    (xa, ya), (xb, yb) = points[i], points[i + 1]
    cx = xa + (xb - xa) * f
    cy = ya + (yb - ya) * f
    size = s0 + (s1 - s0) * u                         # 물체가 보이는 크기
    return cx, cy, size


# ============================================================
# 7. 프레임 그리기 + 정답 박스 기록
# ============================================================
def render_frame(t):
    frame = BACKGROUND.copy()
    boxes = []
    for obj in SCENARIO:
        state = object_state(t, obj)
        if state is None:
            continue
        cx, cy, size = state
        crop = int(round(size * 2))                   # 붙일 사진 크기 = 물체 2배
        sprite = np.asarray(SPRITES[obj[0]].resize((crop, crop), Image.BICUBIC),
                            dtype=np.float32)
        alpha = make_mask(crop)

        # 사진의 왼쪽 위 좌표, 화면 밖으로 나간 부분은 잘라낸다
        x0, y0 = int(round(cx - crop / 2)), int(round(cy - crop / 2))
        fx0, fy0 = max(x0, 0), max(y0, 0)
        fx1, fy1 = min(x0 + crop, W), min(y0 + crop, H)
        if fx1 > fx0 and fy1 > fy0:
            sx0, sy0 = fx0 - x0, fy0 - y0
            sx1, sy1 = sx0 + (fx1 - fx0), sy0 + (fy1 - fy0)
            a = alpha[sy0:sy1, sx0:sx1]
            region = frame[fy0:fy1, fx0:fx1]
            frame[fy0:fy1, fx0:fx1] = region * (1 - a) + sprite[sy0:sy1, sx0:sx1] * a

        # 정답 박스 = 사진 중심 ± 사진크기/4 (물체가 가운데 절반을 차지)
        half = crop / 4
        x1, y1 = max(cx - half, 0), max(cy - half, 0)
        x2, y2 = min(cx + half, W), min(cy + half, H)
        if x2 - x1 >= 8 and y2 - y1 >= 8:            # 화면에 거의 안 보이면 제외
            boxes.append({"id": obj[0], "cls": obj[1],
                          "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]})
    return frame.astype(np.uint8), boxes


# ============================================================
# 8. ffmpeg 로 영상 저장 (프레임을 RGB 그대로 파이프로 보냄)
# ============================================================
OUT_DIR.mkdir(parents=True, exist_ok=True)

ffmpeg = subprocess.Popen(
    ["ffmpeg", "-y", "-loglevel", "error",
     "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
     "-i", "-",
     "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
     str(VIDEO_PATH)],
    stdin=subprocess.PIPE,
)

PREVIEW_FRAMES = [60, 150, 195, 285, 345, 405, 495, 570]   # 각 구간에서 하나씩
preview_images = {}
all_boxes = []

for n in range(N_FRAMES):
    frame, boxes = render_frame(n / FPS)
    ffmpeg.stdin.write(frame.tobytes())
    all_boxes.append(boxes)
    if n in PREVIEW_FRAMES:
        preview_images[n] = (frame, boxes)
    if n % 150 == 0:
        print(f"  {n}/{N_FRAMES} 프레임")

ffmpeg.stdin.close()
ffmpeg.wait()
print(f"영상 저장 : {VIDEO_PATH}")


# ============================================================
# 9. 정답 JSON 저장
# ============================================================
gt = {"fps": FPS, "width": W, "height": H, "classes": CLASSES, "frames": all_boxes}
GT_PATH.write_text(json.dumps(gt, separators=(",", ":")))
print(f"정답 저장 : {GT_PATH}")


# ============================================================
# 10. 미리보기 (정답 박스가 물체에 맞는지 눈으로 확인)
# ============================================================
COLORS = {0: (255, 40, 40), 1: (40, 200, 40)}   # 드론 빨강, 새 초록
TW, TH = 640, 360
sheet = Image.new("RGB", (TW * 4, TH * 2), "white")

for k, n in enumerate(PREVIEW_FRAMES):
    frame, boxes = preview_images[n]
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    for b in boxes:
        draw.rectangle(b["box"], outline=COLORS[b["cls"]], width=4)
        draw.text((b["box"][0] + 4, b["box"][1] + 4),
                  f"id{b['id']} {CLASSES[b['cls']]}", fill=COLORS[b["cls"]])
    draw.text((10, 10), f"frame {n}  ({n / FPS:.1f}s)", fill=(0, 0, 0))
    sheet.paste(img.resize((TW, TH)), ((k % 4) * TW, (k // 4) * TH))

sheet.save(PREVIEW_PATH)
print(f"미리보기 : {PREVIEW_PATH}")
