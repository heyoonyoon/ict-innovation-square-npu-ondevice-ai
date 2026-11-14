"""
c04_track_mac.py — 맥에서 추적기(c04_tracker.py) 테스트

영상 → (가짜 or ONNX) detection → Tracker → 박스/궤적/경보 그리기 → mp4 저장 → GT 와 비교 평가

실행:  python c04_track_mac.py                       (기본 경로)
       python c04_track_mac.py in.mp4 gt.json out.mp4   (경로 바꿔서)
"""
import json
import os
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from c04_tracker import Tracker, iou

# ============================================================
# 0. 설정
# ============================================================
DETECTOR = "onnx"          # "gt_noise" (GT + 노이즈로 흉내) / "onnx" (실제 모델)

HERE = os.path.dirname(os.path.abspath(__file__))
VIDEO_IN = os.path.join(HERE, "video", "c03_test.mp4")
GT_JSON = os.path.join(HERE, "video", "c03_test_gt.json")
VIDEO_OUT = os.path.join(HERE, "video", "c04_tracked.mp4")
if len(sys.argv) == 4:
    VIDEO_IN, GT_JSON, VIDEO_OUT = sys.argv[1:4]

FFMPEG = "/opt/homebrew/bin/ffmpeg"

# 추적기 파라미터
TRACKER_ARGS = dict(iou_threshold=0.3, confirm_frames=3, max_missed=15, history_len=40)

# 경보: drone(cls 0) confirmed track 이 ALERT_HITS 프레임 이상 잡히면 경보
ALERT_HITS = 10
ALERT_HOLD = 15                # 조건이 잠깐(1~2프레임 놓침) 끊겨도 15프레임(0.5초) 동안 경보 유지
DRONE_CLS = 0

# gt_noise 흉내 파라미터
NOISE_JITTER = 0.03            # 박스 크기의 3% 가우시안 흔들림
NOISE_DROP = 0.10              # 10% 확률로 놓침
NOISE_FP = 0.02                # 프레임당 2% 확률로 엉뚱한 박스(오탐)
NOISE_FLIP = 0.05              # 5% 확률로 클래스 뒤바뀜
SEED = 0

CLASS_NAMES = ["drone", "bird", "airplane", "helicopter"]
CLASS_COLORS = [(255, 40, 40), (40, 220, 40), (0, 220, 255), (255, 230, 0)]


# ============================================================
# 1. 영상 읽기/쓰기 (ffmpeg 파이프, cv2 없이)
# ============================================================
def read_frames(path, w, h):
    cmd = [FFMPEG, "-loglevel", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    size = w * h * 3
    while True:
        buf = p.stdout.read(size)
        if len(buf) < size:
            break
        yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    p.wait()


def open_writer(path, w, h, fps):
    cmd = [FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", path]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


# ============================================================
# 2. detection 만들기
# ============================================================
rng = np.random.default_rng(SEED)


def detect_gt_noise(gt_objs, w, h):
    """GT 박스에 흔들림/놓침/오탐/클래스 뒤바뀜을 넣어 detector 흉내"""
    dets = []
    for o in gt_objs:
        if rng.random() < NOISE_DROP:                 # 놓침
            continue
        x1, y1, x2, y2 = o["box"]
        bw, bh = x2 - x1, y2 - y1
        jit = rng.normal(0, NOISE_JITTER, 4) * [bw, bh, bw, bh]
        box = [x1 + jit[0], y1 + jit[1], x2 + jit[2], y2 + jit[3]]
        cls = o["cls"]
        if rng.random() < NOISE_FLIP:                 # 클래스 뒤바뀜
            cls = int(rng.integers(0, len(CLASS_NAMES)))
        dets.append({"box": box, "score": float(rng.uniform(0.5, 0.95)), "cls": cls})
    if rng.random() < NOISE_FP:                       # 오탐: 작은 랜덤 박스
        s = rng.uniform(15, 40)
        x, y = rng.uniform(0, w - s), rng.uniform(0, h - s)
        dets.append({"box": [x, y, x + s, y + s], "score": float(rng.uniform(0.25, 0.4)),
                     "cls": int(rng.integers(0, len(CLASS_NAMES)))})
    return dets


if DETECTOR == "onnx":
    from c04_detector_onnx import detect       # detect(frame_rgb) -> list[Detection]


# ============================================================
# 3. 그리기
# ============================================================
def load_font(size):
    for f in ["/System/Library/Fonts/Supplemental/Arial Bold.ttf",
              "/System/Library/Fonts/Helvetica.ttc",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]:
        if os.path.exists(f):
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


FONT = load_font(18)
FONT_BIG = load_font(56)


def dashed_rect(draw, box, color, width=2, dash=8):
    """점선 사각형: 각 변을 dash 길이로 잘라서 하나 걸러 하나씩 그린다"""
    x1, y1, x2, y2 = box
    for (ax, ay, bx, by) in [(x1, y1, x2, y1), (x2, y1, x2, y2), (x2, y2, x1, y2), (x1, y2, x1, y1)]:
        length = max(abs(bx - ax), abs(by - ay))
        n = max(int(length // dash), 1)
        for i in range(0, n, 2):
            s, e = i / n, min((i + 1) / n, 1.0)
            draw.line([(ax + (bx - ax) * s, ay + (by - ay) * s),
                       (ax + (bx - ax) * e, ay + (by - ay) * e)], fill=color, width=width)


def draw_frame(img, tracks, frame_no, fps, alert):
    draw = ImageDraw.Draw(img)
    W, H = img.size
    n_conf = 0
    for t in tracks:
        if t.state == "tentative":
            draw.rectangle(t.box, outline=(160, 160, 160), width=1)
            continue
        color = CLASS_COLORS[t.cls % len(CLASS_COLORS)]
        n_conf += 1
        if len(t.history) >= 2:                       # 궤적
            draw.line(t.history, fill=color, width=2)
        if t.state == "lost":
            dashed_rect(draw, t.box, color)
        else:
            draw.rectangle(t.box, outline=color, width=3)
        label = f"#{t.id} {CLASS_NAMES[t.cls]} {t.score:.2f}" + (" (lost)" if t.state == "lost" else "")
        tx, ty = t.box[0], max(t.box[1] - 22, 0)
        tb = draw.textbbox((tx, ty), label, font=FONT)
        draw.rectangle(tb, fill=(0, 0, 0))
        draw.text((tx, ty), label, fill=color, font=FONT)

    hud = f"frame {frame_no}  fps {fps:.1f}  tracks {n_conf} (+{len(tracks) - n_conf} tentative)"
    draw.rectangle(draw.textbbox((20, 20), hud, font=FONT), fill=(0, 0, 0))
    draw.text((20, 20), hud, fill=(255, 255, 255), font=FONT)

    if alert:
        draw.rectangle([0, 0, W - 1, H - 1], outline=(255, 0, 0), width=14)
        msg = "DRONE ALERT"
        bb = draw.textbbox((0, 0), msg, font=FONT_BIG)
        x = (W - (bb[2] - bb[0])) // 2
        draw.rectangle([x - 16, 40, x + bb[2] - bb[0] + 16, 40 + bb[3] + 16], fill=(200, 0, 0))
        draw.text((x, 46), msg, fill=(255, 255, 255), font=FONT_BIG)
    return img


# ============================================================
# 4. 경보 규칙
# ============================================================
def alert_condition(tracks):
    # drone 으로 확정된 track 이 ALERT_HITS 이상 잡혔을 때만. bird 등 다른 클래스는 절대 경보 X
    return any(t.state == "confirmed" and t.cls == DRONE_CLS and t.hits >= ALERT_HITS
               for t in tracks)


class Alarm:
    """경보 깜빡임 방지: 마지막으로 조건을 만족한 뒤 ALERT_HOLD 프레임까지 경보 유지"""
    def __init__(self):
        self.left = 0

    def update(self, tracks):
        self.left = ALERT_HOLD if alert_condition(tracks) else max(self.left - 1, 0)
        return self.left > 0


# ============================================================
# 5. 평가 (GT 와 비교)
# ============================================================
class Evaluator:
    def __init__(self):
        self.gt_to_tracks = {}        # GT id → 지금까지 매칭된 track id 집합
        self.gt_frames = {}           # GT id → 등장 프레임 수
        self.gt_matched = 0           # 매칭 성공한 (프레임, GT) 수
        self.gt_total = 0
        self.alarm_drone = self.alarm_nodrone = 0
        self.raw_nodrone = 0          # hold 꼬리를 뺀 '진짜' 오경보 (조건 자체가 참)
        self.frames_drone = self.frames_nodrone = 0
        self.first_drone = self.first_alarm = None

    def add(self, frame_no, gt_objs, tracks, alert, raw):
        confirmed = [t for t in tracks if t.state == "confirmed"]
        # GT 와 confirmed track 을 IoU > 0.5 로 그리디 매칭
        pairs = sorted(((iou(g["box"], t.box), gi, ti) for gi, g in enumerate(gt_objs)
                        for ti, t in enumerate(confirmed)), reverse=True)
        used_g, used_t = set(), set()
        for s, gi, ti in pairs:
            if s <= 0.5 or gi in used_g or ti in used_t:
                continue
            used_g.add(gi); used_t.add(ti)
            self.gt_to_tracks.setdefault(gt_objs[gi]["id"], set()).add(confirmed[ti].id)
        for g in gt_objs:
            self.gt_frames[g["id"]] = self.gt_frames.get(g["id"], 0) + 1
        self.gt_matched += len(used_g)
        self.gt_total += len(gt_objs)

        has_drone = any(g["cls"] == DRONE_CLS for g in gt_objs)
        if has_drone:
            self.frames_drone += 1
            self.alarm_drone += alert
            if self.first_drone is None:
                self.first_drone = frame_no
        else:
            self.frames_nodrone += 1
            self.alarm_nodrone += alert
            self.raw_nodrone += raw
        if alert and self.first_alarm is None:
            self.first_alarm = frame_no

    def report(self, tracker):
        print("\n========== 평가 ==========")
        print(f"GT 객체 수              : {len(self.gt_frames)}")
        print(f"생성된 confirmed ID 수  : {tracker.next_id - 1}")
        switches = 0
        for gid in sorted(self.gt_frames):
            tids = sorted(self.gt_to_tracks.get(gid, []))
            switches += max(len(tids) - 1, 0)
            print(f"  GT {gid:>3} ({self.gt_frames[gid]:>4} 프레임) → track {tids}")
        print(f"ID switch              : {switches}")
        print(f"GT 박스 커버율          : {self.gt_matched}/{self.gt_total}"
              f" = {self.gt_matched / max(self.gt_total, 1):.1%}")
        print(f"경보 ON (GT 에 드론 있음): {self.alarm_drone}/{self.frames_drone} 프레임")
        print(f"경보 ON (GT 에 드론 없음): {self.alarm_nodrone}/{self.frames_nodrone} 프레임"
              f"  (그중 hold 꼬리 제외한 진짜 오경보 {self.raw_nodrone})")
        print(f"첫 드론 등장 / 첫 경보  : frame {self.first_drone} / frame {self.first_alarm}")


# ============================================================
# 6. 메인 루프
# ============================================================
def main():
    gt = json.load(open(GT_JSON))
    W, H, FPS = gt["width"], gt["height"], gt["fps"]
    tracker = Tracker(**TRACKER_ARGS, frame_size=(W, H))
    ev = Evaluator()
    alarm = Alarm()
    writer = open_writer(VIDEO_OUT, W, H, FPS)

    t_prev, fps = time.time(), 0.0
    for i, frame in enumerate(read_frames(VIDEO_IN, W, H)):
        gt_objs = gt["frames"][i] if i < len(gt["frames"]) else []
        if DETECTOR == "onnx":
            dets = detect(frame)
        else:
            dets = detect_gt_noise(gt_objs, W, H)

        tracks = tracker.update(dets)
        alert = alarm.update(tracks)
        ev.add(i, gt_objs, tracks, alert, alert_condition(tracks))

        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6) if i else 0.0  # 처리 fps (지수 평균)
        t_prev = now

        img = draw_frame(Image.fromarray(frame), tracks, i, fps, alert)
        writer.stdin.write(np.asarray(img).tobytes())

    writer.stdin.close()
    writer.wait()
    print(f"저장: {VIDEO_OUT}  ({i + 1} 프레임)")
    ev.report(tracker)


if __name__ == "__main__":
    main()
