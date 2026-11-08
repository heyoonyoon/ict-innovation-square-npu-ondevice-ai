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


# ---------------------------------------------------------
# 1. 설정값
# ---------------------------------------------------------

WORK_DIR = Path.home() / "hailo_models"
DRONE_DIR = WORK_DIR / "drone"

HEF_PATH = DRONE_DIR / "drone_resnet18.hef"

# ImageFolder 알파벳 순서
CLASS_NAMES = ["bird", "drone", "sky"]
DRONE_INDEX = 1

DRONE_THRESHOLD = 0.8      # drone 확률이 이 값 이상이면 "의심 프레임"
ALERT_FRAMES = 5           # 의심 프레임이 연속 몇 번이면 경보

CAMERA_SIZE = (640, 480)   # (가로, 세로)
SHOW_WINDOW = True         # False 면 터미널 출력만


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


# ---------------------------------------------------------
# 3. 전처리 (b05 와 동일)
#    짧은 변 256 Resize → 가운데 224 CenterCrop
#    Normalize 는 하지 않음 (HEF 안에서 함)
# ---------------------------------------------------------

def preprocess(image_rgb):
    h, w = image_rgb.shape[:2]
    ratio = 256 / min(h, w)

    image_256 = cv2.resize(
        image_rgb,
        (round(w * ratio), round(h * ratio))
    )

    h, w = image_256.shape[:2]
    top = (h - input_height) // 2
    left = (w - input_width) // 2

    cropped = image_256[
        top:top + input_height,
        left:left + input_width
    ]

    # HWC → NHWC
    return np.expand_dims(cropped, axis=0).astype(np.uint8)


def softmax(scores):
    exp_scores = np.exp(scores - scores.max())
    return exp_scores / exp_scores.sum()


# ---------------------------------------------------------
# 4. 카메라 시작
#    Picamera2 의 "RGB888" 은 실제 배열 순서가 BGR 이다
#    (OpenCV 와 같은 순서) → b05 처럼 BGR → RGB 변환
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


# ---------------------------------------------------------
# 5. Hailo 장치 설정 (b05 와 동일)
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
    # 6. 실시간 추론 반복
    #    b05 는 infer 를 한 번 했지만
    #    여기서는 while 로 매 프레임 infer
    # -----------------------------------------------------

    with InferVStreams(
        network_group,
        input_vstreams_params,
        output_vstreams_params,
    ) as infer_pipeline:

        with network_group.activate(
            network_group_params
        ):

            drone_streak = 0      # 연속 의심 프레임 수
            frame_count = 0
            fps_start = time.perf_counter()
            fps = 0.0

            try:
                while True:

                    # 6-1. 카메라 프레임
                    frame_bgr = picam2.capture_array()
                    frame_rgb = cv2.cvtColor(
                        frame_bgr,
                        cv2.COLOR_BGR2RGB
                    )

                    # 6-2. 전처리 + NPU 추론
                    input_tensor = preprocess(frame_rgb)

                    infer_start = time.perf_counter()

                    infer_results = infer_pipeline.infer(
                        {input_info.name: input_tensor}
                    )

                    infer_ms = (time.perf_counter() - infer_start) * 1000

                    scores = np.asarray(
                        infer_results[output_info.name]
                    ).reshape(-1)

                    probs = softmax(scores)

                    top_index = int(np.argmax(probs))
                    drone_prob = float(probs[DRONE_INDEX])

                    # 6-3. 연속 프레임 판정
                    if drone_prob >= DRONE_THRESHOLD:
                        drone_streak += 1
                    else:
                        drone_streak = 0

                    alert = drone_streak >= ALERT_FRAMES

                    # 6-4. 전체 FPS (카메라 + 전처리 + 추론 + 출력)
                    frame_count += 1
                    elapsed = time.perf_counter() - fps_start

                    if elapsed >= 1.0:
                        fps = frame_count / elapsed
                        frame_count = 0
                        fps_start = time.perf_counter()

                    # 6-5. 터미널 출력
                    status = "DRONE ALERT" if alert else "-"

                    print(
                        f"{CLASS_NAMES[top_index]:5s} "
                        f"drone={drone_prob:.3f} "
                        f"streak={drone_streak:2d} "
                        f"npu={infer_ms:5.1f}ms "
                        f"fps={fps:4.1f}  "
                        f"{status}"
                    )

                    # 6-6. 영상 창 출력
                    if SHOW_WINDOW:
                        view = frame_bgr.copy()

                        text = (
                            f"{CLASS_NAMES[top_index]} "
                            f"drone={drone_prob:.2f} "
                            f"fps={fps:.1f}"
                        )

                        cv2.putText(
                            view, text, (10, 30),
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
                                view, "DRONE ALERT", (40, 260),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                2.0, (0, 0, 255), 5,
                            )

                        cv2.imshow("Drone Alarm", view)

                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break

            except KeyboardInterrupt:
                print("\n[STOP] Ctrl+C")


# ---------------------------------------------------------
# 7. 정리
# ---------------------------------------------------------

picam2.stop()
cv2.destroyAllWindows()
print("[END]")
