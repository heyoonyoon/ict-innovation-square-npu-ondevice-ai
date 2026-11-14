# =========================================================
# c04_detector_onnx.py  (Mac, onnxruntime)
#
# 하는 일
#   Colab 에서 fine-tune 한 YOLOv8n(4클래스: drone, bird, airplane, helicopter)
#   ONNX 모델로 프레임 한 장을 추론해서 박스 리스트를 돌려준다.
#   다른 코드(트래커, 영상 테스트)가 이 형식을 그대로 쓴다:
#     [{"box": [x1, y1, x2, y2], "score": float, "cls": int}, ...]
#     box 는 "원본 프레임" 픽셀 좌표(float).
#
# 실행
#   npu_env/bin/python DRONE_TRACK/c04_detector_onnx.py            → 단위 테스트만
#   npu_env/bin/python DRONE_TRACK/c04_detector_onnx.py 사진.jpg   → 단위 테스트 + 사진 추론
#   다른 코드에서:  from c04_detector_onnx import detect
#
# 모델이 생기면 꼭 확인할 것
#   1) output/yolov8n_drone.onnx 출력 shape 이 [1, 8, 8400] 인지 (4 + 클래스 4개)
#      → 시작할 때 print 되는 [ONNX OUTPUT] 줄로 확인
#   2) 클래스 순서가 data.yaml 의 names 와 같은지 (0 drone, 1 bird, 2 airplane, 3 helicopter)
#   3) Ultralytics 로 같은 사진을 predict 한 결과와 박스/점수가 거의 같은지
#      (전처리 resize 를 PIL 로 해서 소수점 차이는 조금 날 수 있음)
# =========================================================

import sys
from pathlib import Path

import numpy as np
from PIL import Image


# ---------------------------------------------------------
# 1. 설정값
# ---------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "output" / "yolov8n_drone.onnx"

CLASS_NAMES = ["drone", "bird", "airplane", "helicopter"]
INPUT_SIZE = 640       # export 할 때 imgsz=640
PAD_VALUE = 114        # Ultralytics LetterBox 기본 회색
MAX_DET = 300          # Ultralytics NMS 기본 max_det


_session = None        # 처음 detect() 할 때 만든다 (lazy)
_input_name = None


def _get_session():
    global _session, _input_name

    if _session is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"ONNX 모델 없음: {MODEL_PATH}")

        import onnxruntime as ort

        _session = ort.InferenceSession(
            str(MODEL_PATH),
            providers=["CPUExecutionProvider"],
        )
        _input_name = _session.get_inputs()[0].name

        print("[ONNX INPUT ]", _input_name, _session.get_inputs()[0].shape)
        print("[ONNX OUTPUT]", _session.get_outputs()[0].name, _session.get_outputs()[0].shape)

    return _session


# ---------------------------------------------------------
# 2. 전처리 (Ultralytics LetterBox 와 같은 방식)
#    비율 유지 resize → 남는 곳은 114 로 채워 640x640 (가운데 정렬)
#    RGB, /255, NCHW float32
#    출처: ultralytics/data/augment.py  class LetterBox
#      r = min(640/h, 640/w)
#      new_unpad = round(w*r), round(h*r)
#      top, bottom = round(dh - 0.1), round(dh + 0.1)  (left/right 도 같음)
# ---------------------------------------------------------

def letterbox(image_rgb, size=INPUT_SIZE, pad_value=PAD_VALUE):
    h, w = image_rgb.shape[:2]
    r = min(size / h, size / w)

    new_w = max(round(w * r), 1)
    new_h = max(round(h * r), 1)

    dw = (size - new_w) / 2
    dh = (size - new_h) / 2
    left = round(dw - 0.1)
    top = round(dh - 0.1)

    if (new_w, new_h) != (w, h):
        # Ultralytics 는 cv2.INTER_LINEAR → PIL BILINEAR 로 대신
        resized = np.asarray(
            Image.fromarray(image_rgb).resize((new_w, new_h), Image.BILINEAR)
        )
    else:
        resized = image_rgb

    canvas = np.full((size, size, 3), pad_value, dtype=np.uint8)
    canvas[top:top + new_h, left:left + new_w] = resized

    # (gain, pad_x, pad_y) 는 나중에 박스 되돌릴 때 사용
    return canvas, r, left, top


def preprocess(frame_rgb):
    canvas, r, pad_x, pad_y = letterbox(frame_rgb)

    tensor = canvas.astype(np.float32) / 255.0
    tensor = tensor.transpose(2, 0, 1)[None]   # HWC → NCHW
    return np.ascontiguousarray(tensor), r, pad_x, pad_y


# ---------------------------------------------------------
# 3. 후처리
#    출력 [1, 4+nc, 8400]
#      앞 4줄 = cx, cy, w, h (640 입력 기준 픽셀)
#      나머지 = 클래스 점수 (이미 sigmoid 된 값, objectness 없음)
#    출처: ultralytics/nn/modules/head.py  Detect._inference
#      return torch.cat((dbox, scores.sigmoid()), 1)
# ---------------------------------------------------------

def box_iou_one_to_many(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (box[2] - box[0]) * (box[3] - box[1])
    area_b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(area_a + area_b - inter, 1e-9)


def nms_classwise(boxes, scores, classes, iou_threshold, max_det=MAX_DET):
    """클래스별 NMS. 남길 index 배열(점수 높은 순)을 돌려준다.
    Ultralytics 처럼 클래스마다 좌표를 크게 밀어서(offset) 한 번에 NMS 한다
    → 다른 클래스끼리는 절대 겹치지 않음."""
    if len(boxes) == 0:
        return np.zeros(0, dtype=np.int64)

    offset = classes.astype(np.float32)[:, None] * 7680.0
    shifted = boxes + offset

    order = np.argsort(-scores, kind="stable")
    keep = []

    while order.size > 0 and len(keep) < max_det:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        ious = box_iou_one_to_many(shifted[i], shifted[order[1:]])
        order = order[1:][ious <= iou_threshold]

    return np.array(keep, dtype=np.int64)


def unletterbox(boxes, r, pad_x, pad_y, orig_w, orig_h):
    """640 입력 좌표 → 원본 프레임 좌표 (Ultralytics scale_boxes 와 같은 계산)"""
    boxes = boxes.copy()
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / r
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / r
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, orig_w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, orig_h)
    return boxes


def postprocess(output, r, pad_x, pad_y, orig_w, orig_h,
                conf_threshold, iou_threshold):
    pred = np.asarray(output)

    # [1, 4+nc, 8400] → [8400, 4+nc]  (앞쪽 batch 차원은 떼어냄)
    pred = pred.reshape(pred.shape[-2:])
    if pred.shape[0] < pred.shape[1]:
        pred = pred.T

    xywh = pred[:, :4]
    class_scores = pred[:, 4:]

    cls = class_scores.argmax(axis=1)
    score = class_scores[np.arange(len(cls)), cls]

    mask = score > conf_threshold
    xywh, cls, score = xywh[mask], cls[mask], score[mask]

    if len(score) == 0:
        return []

    # cx,cy,w,h → x1,y1,x2,y2
    boxes = np.empty_like(xywh)
    boxes[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
    boxes[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
    boxes[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
    boxes[:, 3] = xywh[:, 1] + xywh[:, 3] / 2

    keep = nms_classwise(boxes, score, cls, iou_threshold)
    boxes = unletterbox(boxes[keep], r, pad_x, pad_y, orig_w, orig_h)

    return [
        {
            "box": [float(v) for v in b],
            "score": float(s),
            "cls": int(c),
        }
        for b, s, c in zip(boxes, score[keep], cls[keep])
    ]


# ---------------------------------------------------------
# 4. detect() : 외부에서 쓰는 함수
# ---------------------------------------------------------

def detect(frame_rgb, conf_threshold=0.35, iou_threshold=0.5):
    session = _get_session()

    orig_h, orig_w = frame_rgb.shape[:2]
    tensor, r, pad_x, pad_y = preprocess(frame_rgb)

    output = session.run(None, {_input_name: tensor})[0]

    return postprocess(output, r, pad_x, pad_y, orig_w, orig_h,
                       conf_threshold, iou_threshold)


# ---------------------------------------------------------
# 5. 단위 테스트 (가짜 배열, 모델 없이)
# ---------------------------------------------------------

def _self_test():
    # 5-1. letterbox: 1280x720 → r=0.5, 640x360, 위아래 140 씩
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    canvas, r, pad_x, pad_y = letterbox(frame)
    assert canvas.shape == (640, 640, 3)
    assert (r, pad_x, pad_y) == (0.5, 0, 140), (r, pad_x, pad_y)
    # 위 패딩 0~139, 영상 140~499, 아래 패딩 500~639
    assert canvas[139, 0, 0] == 114 and canvas[140, 0, 0] == 0
    assert canvas[499, 0, 0] == 0 and canvas[500, 0, 0] == 114
    # 세로 영상 480x640(가로 480) → r=1, 좌우 80 씩
    _, r2, px2, py2 = letterbox(np.zeros((640, 480, 3), dtype=np.uint8))
    assert (r2, px2, py2) == (1.0, 80, 0)

    # 5-2. 가짜 모델 출력 [1, 8, 8400]
    #   원본(1280x720)의 박스 A drone (100,200)-(300,400) 을 640 공간으로 옮기면
    #   x*0.5, y*0.5+140 → (50,240)-(150,340) → cx=100 cy=290 w=100 h=100
    nc = 4
    out = np.zeros((1, 4 + nc, 8400), dtype=np.float32)

    def put(i, cx, cy, w, h, c, s):
        out[0, :4, i] = [cx, cy, w, h]
        out[0, 4 + c, i] = s

    put(0, 100, 290, 100, 100, 0, 0.90)   # A drone
    put(1, 102, 291, 100, 100, 0, 0.80)   # A 와 거의 겹침, drone → NMS 로 제거
    put(2, 102, 291, 100, 100, 1, 0.70)   # 같은 위치지만 bird → 클래스 달라서 남음
    put(3, 400, 300, 50, 50, 3, 0.20)     # 점수 낮음 → threshold 로 제거
    put(4, 630, 630, 40, 40, 2, 0.60)     # 오른쪽 아래 끝 → clip 확인

    dets = postprocess(out, r, pad_x, pad_y, 1280, 720, 0.35, 0.5)
    print("[TEST] dets:", dets)

    assert len(dets) == 3, dets
    assert [d["cls"] for d in dets] == [0, 1, 2]
    assert np.allclose(dets[0]["box"], [100, 200, 300, 400], atol=1e-4)
    assert abs(dets[0]["score"] - 0.9) < 1e-6
    # 마지막 박스: x2 = (650-0)/0.5 = 1300 → 1280 으로 clip, y2 = (650-140)/0.5=1020 → 720
    assert dets[2]["box"][2] == 1280 and dets[2]["box"][3] == 720
    assert all(isinstance(v, float) for d in dets for v in d["box"])

    # 5-3. NMS 단독: 같은 클래스, IoU 기준 경계
    b = np.array([[0, 0, 10, 10], [0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float32)
    k = nms_classwise(b, np.array([0.5, 0.9, 0.3]), np.array([0, 0, 0]), 0.5)
    assert k.tolist() == [1, 2], k

    # 5-4. 아무것도 없을 때
    assert postprocess(np.zeros((1, 8, 8400), np.float32), r, pad_x, pad_y, 1280, 720, 0.35, 0.5) == []

    print("[TEST] letterbox / NMS / letterbox 되돌리기 OK")


# ---------------------------------------------------------
# 6. 직접 실행
# ---------------------------------------------------------

if __name__ == "__main__":
    _self_test()

    if not MODEL_PATH.exists():
        print(f"\n[INFO] 모델 파일이 아직 없음: {MODEL_PATH}")
        print("       Colab 에서 yolov8n_drone.onnx 를 받아 output/ 에 넣고 다시 실행")
        sys.exit(0)

    if len(sys.argv) < 2:
        print("\n사용법: python c04_detector_onnx.py <이미지 경로>")
        sys.exit(0)

    image = np.asarray(Image.open(sys.argv[1]).convert("RGB"))
    detections = detect(image)

    print(f"\n[{sys.argv[1]}] {image.shape[1]}x{image.shape[0]}  검출 {len(detections)}개")
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        print(f"  {CLASS_NAMES[d['cls']]:10s} {d['score']:.3f}  "
              f"({x1:.0f}, {y1:.0f}) - ({x2:.0f}, {y2:.0f})")
