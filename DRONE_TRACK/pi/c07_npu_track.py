# =========================================================
# c07_npu_track.py  (Raspberry Pi 5 + Hailo-8L)
#
# 하는 일
#   카메라 프레임 → Hailo NPU 로 YOLOv8n(drone/bird/airplane/helicopter) 검출
#   → c04_tracker 로 추적(ID 부여) → 박스·궤적·HUD 를 화면에 그림
#   → 확정된 drone 트랙이 ALERT_HITS 프레임 이상 이어지면 "DRONE ALERT"
#   (경보 화면만, 구동기 제어 없음)
#   b06_npu_alarm.py 를 복사해서 필요한 부분만 바꿈
#
# 실행 (Pi)
#   c04_tracker.py 를 이 파일 옆에 복사해 두고
#   python3 c07_npu_track.py
#   종료: 영상 창에서 q, 또는 터미널에서 Ctrl+C
#
# 실제 HEF 가 생기면 꼭 확인할 것
#   1) 시작할 때 print 되는 [HEF OUTPUT] 이 출력 1개, NMS 형식인지
#      (hailortcli parse-hef yolov8n_drone.hef 로도 확인 가능,
#       보통 이름 "yolov8n/yolov8_nms_postprocess", 클래스 4, 클래스당 최대 100)
#   2) 클래스 순서 0 drone, 1 bird, 2 airplane, 3 helicopter (data.yaml 과 같음)
#   3) 박스가 실제 물체 위치에 맞는지 (letterbox 되돌리기가 맞는지)
#      → 어긋나면 LETTERBOX = False 로 바꿔 비교
#   4) HEF 안의 NMS 점수 기준(yolov8n_nms_config.json 의 nms_scores_th=0.2)보다
#      CONF_THRESHOLD 가 낮으면 의미 없음
# =========================================================

import time
from pathlib import Path

import cv2
import numpy as np
from picamera2 import Picamera2

from hailo_platform import (
    HEF,
    VDevice,
    ConfigureParams,
    HailoStreamInterface,
    InputVStreamParams,
    OutputVStreamParams,
    InferVStreams,
    FormatType,
)

from c04_tracker import Tracker


# ---------------------------------------------------------
# 1. 설정값
# ---------------------------------------------------------

WORK_DIR = Path.home() / "hailo_models"
DRONE_DIR = WORK_DIR / "drone_track"

HEF_PATH = DRONE_DIR / "yolov8n_drone.hef"

# Colab 학습 data.yaml 의 names 순서
CLASS_NAMES = ["drone", "bird", "airplane", "helicopter"]
DRONE_INDEX = 0

# 클래스별 색 (BGR)
CLASS_COLORS = {
    0: (0, 0, 255),       # drone      빨강
    1: (0, 200, 0),       # bird       초록
    2: (255, 200, 0),     # airplane   하늘색(cyan 계열)
    3: (0, 220, 255),     # helicopter 노랑
}
TENTATIVE_COLOR = (160, 160, 160)   # 아직 확정 안 된 트랙: 회색

CONF_THRESHOLD = 0.35      # 이 점수 이상 검출만 트래커로
ALERT_HITS = 10            # 확정 drone 트랙이 이만큼 검출되면 경보

# 트래커 설정 (c04_tracker.py)
TRACK_IOU = 0.3
CONFIRM_FRAMES = 3
MAX_MISSED = 15

CAMERA_SIZE = (1280, 720)  # (가로, 세로)  느리면 (640, 480)
SHOW_WINDOW = True         # False 면 터미널 출력만
WINDOW_SIZE = (960, 540)   # 처음 창 크기 (마우스로 늘리고 줄일 수 있음)
PRINT_EVERY = 10           # 터미널 출력은 N 프레임마다

# 전처리 방식
#   True  = letterbox (비율 유지 + 114 회색 패딩)  ← Model Zoo yolov8 기본
#   False = 그냥 640x640 으로 늘리기
LETTERBOX = True
PAD_VALUE = 114


if not HEF_PATH.exists():
    raise FileNotFoundError(f"HEF 파일 없음: {HEF_PATH}")


# ---------------------------------------------------------
# 2. HEF 로드
# ---------------------------------------------------------

hef = HEF(str(HEF_PATH))

input_info = hef.get_input_vstream_infos()[0]
output_info = hef.get_output_vstream_infos()[0]

input_height, input_width, input_channels = input_info.shape

print("[HEF INPUT ]", input_info.name, input_info.shape)
print("[HEF OUTPUT]", output_info.name, output_info.shape)

if len(hef.get_output_vstream_infos()) != 1:
    print("[WARN] 출력이 1개가 아님 → NMS 가 HEF 에 안 들어갔을 수 있음:",
          [o.name for o in hef.get_output_vstream_infos()])


# ---------------------------------------------------------
# 3. 전처리
#    Hailo Model Zoo 의 yolov8n 은 Ultralytics 와 같은 letterbox 를 씀
#      cfg/base/yolo.yaml : preprocessing.meta_arch = yolo_v5, padding_color = 114
#      cfg/base/base.yaml : preprocessing.centered = true
#      core/preprocessing/detection_preprocessing.py : yolo_v5() → letterbox()
#        ratio = min(H/h, W/w), new = int(w*ratio), int(h*ratio)
#        top = round(dh - 0.1), left = round(dw - 0.1)
#        cv2.INTER_AREA (줄일 때) / INTER_LINEAR (키울 때)
#      (calib/ 이미지도 hailomz 가 이 방식으로 만들어 양자화함)
#    /255 정규화는 HEF 안에 있음 (yolov8n.alls 의 normalization([0,0,0],[255,255,255]))
#    → 여기서는 uint8 RGB 그대로 넣는다
#    https://github.com/hailo-ai/hailo_model_zoo/tree/v2.17/hailo_model_zoo/cfg
# ---------------------------------------------------------

def preprocess(image_rgb):
    h, w = image_rgb.shape[:2]

    if not LETTERBOX:
        resized = cv2.resize(image_rgb, (input_width, input_height))
        # (x 배율, y 배율, pad_x, pad_y)
        info = (input_width / w, input_height / h, 0, 0)
        return np.expand_dims(resized, axis=0).astype(np.uint8), info

    ratio = min(input_height / h, input_width / w)
    new_w = int(w * ratio)
    new_h = int(h * ratio)

    left = round((input_width - new_w) / 2 - 0.1)
    top = round((input_height - new_h) / 2 - 0.1)

    resized = cv2.resize(
        image_rgb,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA if ratio < 1.0 else cv2.INTER_LINEAR,
    )

    canvas = np.full(
        (input_height, input_width, 3), PAD_VALUE, dtype=np.uint8
    )
    canvas[top:top + new_h, left:left + new_w] = resized

    # HWC → NHWC
    return np.expand_dims(canvas, axis=0), (ratio, ratio, left, top)


# ---------------------------------------------------------
# 4. 후처리: Hailo NMS 출력 → 공통 검출 형식
#
#    HailoRT 4.23 InferVStreams.infer() 의 NMS 출력 (FLOAT32, tf_nms_format=False)
#      results[name] = [ 프레임0, 프레임1, ... ]           ← batch 리스트
#      프레임        = [ 클래스0 배열, 클래스1 배열, ... ]  ← 클래스 수만큼
#      클래스 배열   = np.ndarray (N, 5)  N=그 클래스 박스 수 (0 이면 (0,5))
#      한 줄         = [y_min, x_min, y_max, x_max, score]  0~1 (640 입력 기준)
#    출처: hailort v4.23.0 pyhailort.py
#            InferVStreams.infer → output_raw_buffer_to_nms_format(_single_frame)
#          hailort.h  hailo_bbox_float32_t {y_min, x_min, y_max, x_max, score}
#
#    다른 모양이 와도 버티도록:
#      - batch 를 뺀 [클래스 배열...] 리스트
#      - tf_nms_format=True 형식 ndarray [batch, 클래스, 5, max_box] (빈칸은 점수 0)
#      - ndarray [클래스, N, 5]
#
#    결과 box 는 원본 프레임 픽셀 [x1, y1, x2, y2]
#    pre_info = preprocess 가 돌려준 (sx, sy, pad_x, pad_y),
#               None 이면 그냥 늘리기(plain resize)로 보고 되돌림
# ---------------------------------------------------------

def _per_class_arrays(raw):
    """여러 모양의 NMS 출력을 '클래스별 (N,5) 배열 리스트' 하나로 맞춘다"""

    # ndarray 로 온 경우 (tf 형식 등)
    if isinstance(raw, np.ndarray) and raw.dtype != object:
        arr = raw
        if arr.ndim == 4:              # [batch, 클래스, ?, ?] → 첫 프레임
            arr = arr[0]
        if arr.ndim == 3:
            # tf 형식 [클래스, 5, max_box] → [클래스, max_box, 5]
            if arr.shape[-1] != 5 and arr.shape[-2] == 5:
                arr = np.swapaxes(arr, -1, -2)
            return [arr[c] for c in range(arr.shape[0])]
        if arr.ndim == 2 and arr.shape[-1] == 5:   # 클래스 1개짜리
            return [arr]
        raise ValueError(f"알 수 없는 NMS 출력 shape: {arr.shape}")

    items = list(raw)
    if len(items) == 0:
        return []

    # [프레임0] 처럼 batch 로 한 번 더 감싸져 있으면 벗긴다
    first = items[0]
    if len(items) == 1 and isinstance(first, (list, tuple)):
        items = list(first)
    elif len(items) == 1 and isinstance(first, np.ndarray) and first.dtype == object:
        items = list(first)
    elif (len(items) == 1 and isinstance(first, np.ndarray)
          and first.ndim == 3):
        return _per_class_arrays(first)

    return [np.asarray(c, dtype=np.float32).reshape(-1, 5) for c in items]


def parse_hailo_nms(raw, frame_w, frame_h, conf_threshold,
                    pre_info=None, input_w=640, input_h=640):
    if pre_info is None:
        pre_info = (input_w / frame_w, input_h / frame_h, 0, 0)
    sx, sy, pad_x, pad_y = pre_info

    detections = []

    for cls_id, boxes in enumerate(_per_class_arrays(raw)):
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 5)
        if boxes.size == 0:
            continue

        for y1, x1, y2, x2, score in boxes:
            if score < conf_threshold:
                continue

            # 0~1 → 640 입력 픽셀 → 패딩 빼기 → 배율 나누기 → 원본 픽셀
            bx1 = (x1 * input_w - pad_x) / sx
            by1 = (y1 * input_h - pad_y) / sy
            bx2 = (x2 * input_w - pad_x) / sx
            by2 = (y2 * input_h - pad_y) / sy

            bx1 = min(max(bx1, 0.0), frame_w)
            bx2 = min(max(bx2, 0.0), frame_w)
            by1 = min(max(by1, 0.0), frame_h)
            by2 = min(max(by2, 0.0), frame_h)

            if bx2 <= bx1 or by2 <= by1:
                continue

            detections.append({
                "box": [float(bx1), float(by1), float(bx2), float(by2)],
                "score": float(score),
                "cls": int(cls_id),
            })

    detections.sort(key=lambda d: d["score"], reverse=True)
    return detections


# ---------------------------------------------------------
# 5. 그리기
#    tentative = 얇은 회색
#    confirmed = 클래스 색 굵은 박스 + "#id drone 0.87"
#    lost      = 같은 색 얇은 점선
#    궤적(history) = 선
# ---------------------------------------------------------

def draw_dashed_rect(img, p1, p2, color, thickness=1, dash=8):
    x1, y1 = p1
    x2, y2 = p2
    for x in range(x1, x2, dash * 2):
        cv2.line(img, (x, y1), (min(x + dash, x2), y1), color, thickness)
        cv2.line(img, (x, y2), (min(x + dash, x2), y2), color, thickness)
    for y in range(y1, y2, dash * 2):
        cv2.line(img, (x1, y), (x1, min(y + dash, y2)), color, thickness)
        cv2.line(img, (x2, y), (x2, min(y + dash, y2)), color, thickness)


def draw_track(img, track):
    x1, y1, x2, y2 = [int(round(v)) for v in track.box]
    color = CLASS_COLORS.get(track.cls, (255, 255, 255))

    if track.state == "tentative":
        cv2.rectangle(img, (x1, y1), (x2, y2), TENTATIVE_COLOR, 1)
        return

    # 궤적
    if len(track.history) >= 2:
        pts = np.array(
            [[int(cx), int(cy)] for cx, cy in track.history], dtype=np.int32
        )
        cv2.polylines(img, [pts], False, color, 2)

    if track.state == "lost":
        draw_dashed_rect(img, (x1, y1), (x2, y2), color, 1)
        return

    # confirmed
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    track_id = "?" if track.id is None else track.id
    name = CLASS_NAMES[track.cls] if 0 <= track.cls < len(CLASS_NAMES) else str(track.cls)
    label = f"#{track_id} {name} {track.score:.2f}"

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    ty = max(y1, th + 6)
    cv2.rectangle(img, (x1, ty - th - 6), (x1 + tw + 4, ty), color, -1)
    cv2.putText(img, label, (x1 + 2, ty - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


# ---------------------------------------------------------
# 6. 카메라 시작
#    Picamera2 의 "RGB888" 은 실제 배열 순서가 BGR 이다
#    (OpenCV 와 같은 순서) → 추론용으로 BGR → RGB 변환
# ---------------------------------------------------------

picam2 = Picamera2()

camera_config = picam2.create_preview_configuration(
    main={"size": CAMERA_SIZE, "format": "RGB888"}
)

picam2.configure(camera_config)
picam2.start()
time.sleep(2)   # 자동초점, 밝기 맞추는 시간

print("\n[CAMERA] started", CAMERA_SIZE)
print("종료: 터미널에서 Ctrl+C, 또는 영상 창에서 q")

# 창 크기 조절 가능하게 (기본값은 영상 크기로 고정이라 화면보다 크면 못 줄임)
if SHOW_WINDOW:
    cv2.namedWindow("Drone Track", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Drone Track", *WINDOW_SIZE)

tracker = Tracker(
    iou_threshold=TRACK_IOU,
    confirm_frames=CONFIRM_FRAMES,
    max_missed=MAX_MISSED,
    frame_size=CAMERA_SIZE,   # 화면 밖으로 나간 물체는 바로 지움 (번호 이어받기 방지)
)


# ---------------------------------------------------------
# 7. Hailo 장치 설정 (b06 과 동일)
#    출력 FLOAT32 → NMS 출력이 0~1 float 박스로 나옴
# ---------------------------------------------------------

with VDevice() as target:

    configure_params = ConfigureParams.create_from_hef(
        hef=hef,
        interface=HailoStreamInterface.PCIe,
    )

    network_groups = target.configure(
        hef,
        configure_params,
    )

    network_group = network_groups[0]

    network_group_params = network_group.create_params()

    input_vstreams_params = InputVStreamParams.make(
        network_group,
        format_type=FormatType.UINT8,
    )

    output_vstreams_params = OutputVStreamParams.make(
        network_group,
        format_type=FormatType.FLOAT32,
    )


    # -----------------------------------------------------
    # 8. 실시간 추론 + 추적 반복
    # -----------------------------------------------------

    with InferVStreams(
        network_group,
        input_vstreams_params,
        output_vstreams_params,
    ) as infer_pipeline:

        with network_group.activate(
            network_group_params
        ):

            frame_index = 0
            frame_count = 0
            fps_start = time.perf_counter()
            fps = 0.0

            try:
                while True:

                    # 8-1. 카메라 프레임
                    frame_bgr = picam2.capture_array()
                    frame_rgb = cv2.cvtColor(
                        frame_bgr,
                        cv2.COLOR_BGR2RGB
                    )
                    frame_h, frame_w = frame_bgr.shape[:2]

                    # 8-2. 전처리 + NPU 추론
                    input_tensor, pre_info = preprocess(frame_rgb)

                    infer_start = time.perf_counter()

                    infer_results = infer_pipeline.infer(
                        {input_info.name: input_tensor}
                    )

                    infer_ms = (time.perf_counter() - infer_start) * 1000

                    # 8-3. NMS 출력 → 검출 리스트
                    detections = parse_hailo_nms(
                        infer_results[output_info.name],
                        frame_w, frame_h,
                        CONF_THRESHOLD,
                        pre_info=pre_info,
                        input_w=input_width,
                        input_h=input_height,
                    )

                    # 8-4. 추적
                    tracks = tracker.update(detections)

                    confirmed = [t for t in tracks if t.state == "confirmed"]

                    alert_tracks = [
                        t for t in confirmed
                        if t.cls == DRONE_INDEX and t.hits >= ALERT_HITS
                    ]
                    alert = len(alert_tracks) > 0

                    # 8-5. 전체 FPS (카메라 + 전처리 + 추론 + 추적 + 출력)
                    frame_index += 1
                    frame_count += 1
                    elapsed = time.perf_counter() - fps_start

                    if elapsed >= 1.0:
                        fps = frame_count / elapsed
                        frame_count = 0
                        fps_start = time.perf_counter()

                    # 8-6. 터미널 출력
                    if alert or frame_index % PRINT_EVERY == 0:
                        status = "DRONE ALERT" if alert else "-"
                        names = " ".join(
                            f"#{t.id}:{CLASS_NAMES[t.cls]}" for t in confirmed
                        )
                        print(
                            f"det={len(detections):2d} "
                            f"tracks={len(confirmed):2d} "
                            f"npu={infer_ms:5.1f}ms "
                            f"fps={fps:4.1f}  "
                            f"{status}  {names}"
                        )

                    # 8-7. 영상 창 출력
                    if SHOW_WINDOW:
                        view = frame_bgr.copy()

                        for t in tracks:
                            draw_track(view, t)

                        hud = (
                            f"fps={fps:.1f} npu={infer_ms:.1f}ms "
                            f"tracks={len(confirmed)} det={len(detections)}"
                        )
                        cv2.putText(
                            view, hud, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 0), 4,
                        )
                        cv2.putText(
                            view, hud, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (255, 255, 255), 2,
                        )

                        if alert:
                            # 빨간 테두리 + 경보 글씨 (BGR 이라 빨강 = (0,0,255))
                            cv2.rectangle(
                                view, (0, 0),
                                (view.shape[1] - 1, view.shape[0] - 1),
                                (0, 0, 255), 12,
                            )
                            cv2.putText(
                                view, "DRONE ALERT",
                                (40, view.shape[0] // 2),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                2.0, (0, 0, 255), 5,
                            )

                        cv2.imshow("Drone Track", view)

                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break

            except KeyboardInterrupt:
                print("\n[STOP] Ctrl+C")


# ---------------------------------------------------------
# 9. 정리
# ---------------------------------------------------------

picam2.stop()
cv2.destroyAllWindows()
print("[END]")
