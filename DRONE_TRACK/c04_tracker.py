"""
c04_tracker.py — 아주 작은 다중 객체 추적기 (SORT 스타일)

- numpy 만 사용 (라즈베리파이에서도 그대로 동작)
- 흐름: Kalman 예측 → IoU 매칭(그리디) → 매칭된 track 갱신 / 새 track 생성 / 놓친 track 처리

Detection = {"box": [x1, y1, x2, y2], "score": float, "cls": int}

Track.id 규칙:
- tentative(후보) 상태에서는 id = None
- confirmed 가 되는 순간 1, 2, 3 ... 순서대로 화면 표시용 id 를 받는다
- lost → confirmed 로 돌아와도 id 는 그대로 유지
"""
import numpy as np


# ============================================================
# 1. IoU (두 박스가 얼마나 겹치는지 0~1)
# ============================================================
def iou(box_a, box_b):
    # 겹치는 영역(교집합)의 좌상단/우하단
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    # 안 겹치면 폭/높이가 음수 → 0 으로 자른다
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter == 0:
        return 0.0

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter     # 합집합 = 두 넓이 합 - 교집합
    return inter / union


# ============================================================
# 2. 매칭 (track ↔ detection 짝 찾기, 그리디)
# ============================================================
def match(tracks, detections, iou_threshold=0.3):
    """반환: (matches[(t_idx, d_idx)], 남은 track idx, 남은 det idx)

    1순위: IoU >= iou_threshold 인 짝 (IoU 큰 순)
    2순위(빠른 움직임 대비): IoU 가 부족해도 중심 거리 < max(track w, h) 이고
          크기가 비슷하면(긴 변 비율 0.5~2배) 허용. 크기 조건이 없으면 큰 박스 옆의
          작은 오탐을 잘못 물어버린다. 점수를 음수(-거리비율)로 줘서 항상 IoU 짝보다 뒤.
    클래스는 비교하지 않는다 (drone/bird 가 초반에 뒤바뀔 수 있어서).
    """
    candidates = []                                   # (점수, t_idx, d_idx)
    for ti, t in enumerate(tracks):
        limit = max(t.box[2] - t.box[0], t.box[3] - t.box[1])   # 허용 중심 거리
        for di, d in enumerate(detections):
            s = iou(t.box, d["box"])
            if s >= iou_threshold:
                candidates.append((s, ti, di))
                continue
            (tx, ty), (dx, dy) = center(t.box), center(d["box"])
            dist = np.hypot(tx - dx, ty - dy)
            size = max(d["box"][2] - d["box"][0], d["box"][3] - d["box"][1])
            if dist < limit and 0.5 < size / limit < 2.0:
                candidates.append((-dist / limit, ti, di))

    candidates.sort(key=lambda c: c[0], reverse=True)  # 점수 높은 순
    matches, used_t, used_d = [], set(), set()
    for s, ti, di in candidates:
        if ti in used_t or di in used_d:              # 이미 짝이 있으면 건너뜀
            continue
        matches.append((ti, di))
        used_t.add(ti)
        used_d.add(di)

    un_t = [i for i in range(len(tracks)) if i not in used_t]
    un_d = [i for i in range(len(detections)) if i not in used_d]
    return matches, un_t, un_d


# ============================================================
# 3. 박스 ↔ (cx, cy, w, h) 변환 도우미
# ============================================================
def center(box):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def box_to_cxcywh(box):
    cx, cy = center(box)
    return np.array([cx, cy, box[2] - box[0], box[3] - box[1]], dtype=float)


def cxcywh_to_box(z):
    cx, cy, w, h = z
    w, h = max(w, 1.0), max(h, 1.0)                   # 크기가 0/음수가 되지 않게
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


# ============================================================
# 4. Kalman 필터 (등속 모델)
# ============================================================
class KalmanBox:
    """상태 x = [cx, cy, w, h, vx, vy, vw, vh]  (v = 1프레임당 변화량)
    측정 z = [cx, cy, w, h]  (detector 가 주는 박스)
    """

    def __init__(self, box):
        self.x = np.zeros(8)
        self.x[:4] = box_to_cxcywh(box)               # 속도는 0 으로 시작

        # F: 상태 전이.  다음 위치 = 지금 위치 + 속도 (dt = 1프레임)
        self.F = np.eye(8)
        for i in range(4):
            self.F[i, i + 4] = 1.0

        # H: 상태에서 측정값(앞 4개)만 꺼내는 행렬
        self.H = np.eye(4, 8)

        # P: 불확실성. 속도는 처음에 전혀 모르니 크게
        s = max(self.x[2], self.x[3])
        self.P = np.diag([s / 10, s / 10, s / 10, s / 10, s, s, s / 4, s / 4]) ** 2

    def _noise(self):
        # 박스 크기에 비례하는 잡음 (큰 박스는 픽셀 오차도 크다)
        s = max(self.x[2], self.x[3], 1.0)
        Q = np.diag([s / 20, s / 20, s / 40, s / 40,      # 위치/크기 흔들림
                     s / 10, s / 10, s / 80, s / 80]) ** 2  # 속도 변화(가속) 허용량
        R = np.diag([s / 20, s / 20, s / 10, s / 10]) ** 2  # 측정(detector) 오차
        return Q, R

    def predict(self):
        Q, _ = self._noise()
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + Q
        return cxcywh_to_box(self.x[:4])

    def update(self, box):
        _, R = self._noise()
        z = box_to_cxcywh(box)
        y = z - self.H @ self.x                       # 잔차: 측정 - 예측
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)      # Kalman gain
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P

    @property
    def velocity(self):
        return (float(self.x[4]), float(self.x[5]))


# ============================================================
# 5. Track (객체 하나)
# ============================================================
class Track:
    VOTE_LEN = 10                                     # 클래스 투표에 쓰는 최근 프레임 수

    def __init__(self, det, history_len):
        self.id = None                                # confirmed 될 때 부여
        self.kf = KalmanBox(det["box"])
        self.box = list(det["box"])
        self.score = det["score"]
        self.cls_votes = [det["cls"]]
        self.cls = det["cls"]
        self.state = "tentative"
        self.hits = 1                                 # 매칭된 총 프레임 수
        self.missed = 0                               # 연속으로 놓친 프레임 수
        self.history = []                             # 중심점 궤적 (update 끝에서 추가)
        self.history_len = history_len

    @property
    def velocity(self):
        return self.kf.velocity

    def add_history(self):
        self.history.append(center(self.box))
        if len(self.history) > self.history_len:
            self.history.pop(0)

    def hit(self, det):
        self.kf.update(det["box"])
        self.box = list(det["box"])
        self.score = det["score"]
        self.cls_votes = (self.cls_votes + [det["cls"]])[-self.VOTE_LEN:]
        self.cls = max(set(self.cls_votes), key=self.cls_votes.count)  # 다수결
        self.hits += 1
        self.missed = 0


# ============================================================
# 6. Tracker (매 프레임 update 호출)
# ============================================================
class Tracker:
    def __init__(self, iou_threshold=0.3, confirm_frames=3, max_missed=15, history_len=40,
                 frame_size=None):
        # frame_size=(W, H) 를 주면: lost track 의 예측 중심이 화면 밖으로 나가면 바로 삭제
        # (화면 밖으로 나간 드론의 id 를, 같은 가장자리로 들어온 새 새(bird)가 이어받는 것 방지)
        self.frame_size = frame_size
        self.iou_threshold = iou_threshold
        self.confirm_frames = confirm_frames
        self.max_missed = max_missed
        self.history_len = history_len
        self.tracks = []
        self.next_id = 1                              # 다음 confirmed id

    def _outside(self, t):
        if self.frame_size is None:
            return False
        cx, cy = center(t.box)
        return not (0 <= cx < self.frame_size[0] and 0 <= cy < self.frame_size[1])

    def update(self, detections):
        # (1) 모든 track 위치를 Kalman 으로 한 프레임 앞으로 예측
        for t in self.tracks:
            t.box = t.kf.predict()

        # (2) 예측 박스 ↔ detection 매칭
        matches, un_t, un_d = match(self.tracks, detections, self.iou_threshold)

        # (3) 매칭 성공: 갱신, 조건 되면 confirmed
        for ti, di in matches:
            t = self.tracks[ti]
            t.hit(detections[di])
            if t.state == "tentative" and t.hits >= self.confirm_frames:
                t.state = "confirmed"
                t.id = self.next_id
                self.next_id += 1
            elif t.state == "lost":
                t.state = "confirmed"                 # 같은 id 로 복귀

        # (4) 매칭 실패한 track: tentative 는 바로 삭제, confirmed 는 lost
        dead = set()
        for ti in un_t:
            t = self.tracks[ti]
            t.missed += 1
            if t.state == "tentative" or t.missed >= self.max_missed or self._outside(t):
                dead.add(ti)
            else:
                t.state = "lost"                      # box 는 Kalman 예측값 유지
        self.tracks = [t for i, t in enumerate(self.tracks) if i not in dead]

        # (5) 남은 detection 은 새 tentative track
        for di in un_d:
            self.tracks.append(Track(detections[di], self.history_len))

        for t in self.tracks:
            t.add_history()
        return list(self.tracks)
