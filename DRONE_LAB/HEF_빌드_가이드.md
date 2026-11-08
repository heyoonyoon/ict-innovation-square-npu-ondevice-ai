# 드론 분류 모델 HEF 빌드 가이드

이 문서는 맥북에서 만든 드론 분류 ONNX 모델을 Windows PC의 Ubuntu VM에서 Hailo-8L용 HEF로 바꾸는 순서를 정리한 것이다. 수업 Step 9에서 mobilenet_v1로 했던 순서를 그대로 따르고, 직접 학습한 모델을 넣기 위해 바뀌는 부분만 따로 표시했다. 다른 모델을 HEF로 바꿀 때도 마지막 절의 표에 있는 값만 바꾸면 같은 순서로 다시 할 수 있다.


## 1. 전체 흐름

모델은 네 곳을 거쳐 라즈베리파이까지 간다. 맥북에서 학습과 ONNX 변환을 마치고, Windows의 공유 폴더를 거쳐 Ubuntu VM 안의 Hailo Docker 컨테이너에서 HEF를 만든 뒤, 다시 공유 폴더로 꺼내 라즈베리파이로 옮긴다.

```
맥북  b02 학습 → b03 ONNX 변환, b04_make_calib 으로 calib 100장 준비
  ↓  (USB 등으로 복사)
Windows  C:\NPU_SHARE\models, C:\NPU_SHARE\calib\drone
  ↓  (VirtualBox 공유 폴더 /media/sf_NPU_SHARE)
Ubuntu VM  ~/hailo_build/shared_with_docker
  ↓  (Docker 마운트 /local/shared_with_docker)
Hailo 컨테이너  parse → optimize → compile → HEF
  ↓
Windows  C:\NPU_SHARE\output\drone_resnet18.hef → 라즈베리파이
```

Hailo 컨테이너 안에서는 세 단계를 거친다. parse는 ONNX를 Hailo 내부 표현인 HAR(Hailo Archive) 파일로 옮기고, optimize는 calib 사진으로 각 층의 값 범위를 재서 INT8로 양자화하며, compile은 양자화된 HAR을 Hailo-8L 칩에서 실행되는 HEF 파일로 만든다.


## 2. 파일 위치

맥북에서 Windows로 옮길 파일은 두 개이고, 각각 아래 자리에 둔다.

| 파일 | 맥북 위치 | Windows 위치 | Ubuntu에서 보이는 경로 |
|---|---|---|---|
| ONNX 모델 | `DRONE_LAB/output/b03_resnet18_drone.onnx` | `C:\NPU_SHARE\models\` | `/media/sf_NPU_SHARE/models/` |
| calib 사진 100장 | `DRONE_LAB/calib/drone/` | `C:\NPU_SHARE\calib\drone\` | `/media/sf_NPU_SHARE/calib/drone/` |

Ubuntu와 컨테이너 안에서는 아래 폴더를 쓴다. 수업 때 만든 `C10`, `step9` 폴더와 섞이지 않도록 드론 작업은 `DRONE` 폴더에서만 한다.

| 용도 | Ubuntu 경로 | 컨테이너 안 경로 |
|---|---|---|
| 작업 폴더 | `~/hailo_build/shared_with_docker/DRONE` | `/local/shared_with_docker/DRONE` |
| ONNX | `~/hailo_build/shared_with_docker/models` | `/local/shared_with_docker/models` |
| calib | `~/hailo_build/shared_with_docker/calib/drone` | `/local/shared_with_docker/calib/drone` |
| 결과 HEF | `~/hailo_build/shared_with_docker/output` | `/local/shared_with_docker/output` |


## 3. Ubuntu에서 파일 준비

먼저 공유 폴더에 두 파일이 제대로 들어왔는지 확인한다. 두 번째 명령의 결과가 100이어야 한다.

```bash
ls -lh /media/sf_NPU_SHARE/models/b03_resnet18_drone.onnx
find /media/sf_NPU_SHARE/calib/drone -type f | wc -l
```

작업 폴더를 만들고 두 파일을 복사한다.

```bash
mkdir -p ~/hailo_build/shared_with_docker/DRONE
mkdir -p ~/hailo_build/shared_with_docker/calib/drone

cp -v /media/sf_NPU_SHARE/models/b03_resnet18_drone.onnx ~/hailo_build/shared_with_docker/models/
cp -v /media/sf_NPU_SHARE/calib/drone/* ~/hailo_build/shared_with_docker/calib/drone/
```

복사가 끝나면 반드시 권한을 연다. Ubuntu에서 `mkdir`로 만든 폴더는 기본 권한이 775라서 컨테이너 안의 hailo 사용자가 쓰기를 못 하고, 이 상태로 parse를 하면 HAR 파일을 저장하는 순간 `Permission denied`로 멈춘다. 수업 실습 때 `C10` 폴더에서 이 오류가 났던 이유가 이것이다.

```bash
chmod -R 777 ~/hailo_build/shared_with_docker
ls -ld ~/hailo_build/shared_with_docker/DRONE
find ~/hailo_build/shared_with_docker/calib/drone -type f | wc -l
```

`ls -ld` 결과가 `drwxrwxrwx`로 시작하고 마지막 명령이 100을 출력하면 준비가 끝난 것이다.


## 4. 컨테이너 들어가기

수업 때 이미 컨테이너를 만들어 두었으므로 `--resume`으로 기존 컨테이너에 다시 들어간다.

```bash
cd ~/hailo_build
./hailo_ai_sw_suite_docker_run.sh --resume
```

옵션 없이 실행하면 `Can't start a new container, already found one`이라는 안내가 나온다. 이때 `--override`를 쓰면 기존 컨테이너를 지우고 새로 만드는데, 컨테이너 안에만 있던 파일은 사라진다. 작업 파일은 모두 `shared_with_docker` 아래에 있어서 지워지지 않지만, 평소에는 `--resume`을 쓴다.

컨테이너에 들어가면 작업 폴더로 이동해 쓰기가 되는지 먼저 확인한다. 아무 오류 없이 다음 줄로 넘어가면 정상이다.

```bash
cd /local/shared_with_docker/DRONE
touch write_test.txt && rm write_test.txt
```


## 5. Parse

수업에서는 Model Zoo가 가진 mobilenet_v1을 그대로 불러왔지만, 이번에는 직접 학습한 모델을 넣어야 한다. 그래서 Model Zoo에 등록된 같은 구조의 모델인 `resnet_v1_18`의 설정을 빌려 쓰고, 모델 파일만 `--ckpt` 옵션으로 직접 만든 ONNX로 바꿔 끼운다.

먼저 Model Zoo에 `resnet_v1_18`이 있는지 확인한다.

```bash
hailomz info resnet_v1_18
```

정보가 출력되면 parse를 실행한다. `--start-node-names`와 `--end-node-names`에는 b03 로그에 찍힌 ONNX의 입력 이름 `input`과 출력 이름 `logits`를 넣는다. 이 두 값을 지정하지 않으면 Model Zoo 원래 모델의 노드 이름을 찾다가 실패할 수 있다.

```bash
hailomz parse resnet_v1_18 --hw-arch hailo8l \
  --ckpt /local/shared_with_docker/models/b03_resnet18_drone.onnx \
  --start-node-names input --end-node-names logits
```

로그 끝에 `Saved HAR to: /local/shared_with_docker/DRONE/resnet_v1_18.har`가 나오면 성공이다. 수업 때처럼 parse 결과를 따로 보관한다.

```bash
ls -lh
cp resnet_v1_18.har drone_parsed.har
```


## 6. Optimize

VM에는 GPU가 없으므로 CPU로만 계산하도록 설정한 뒤 optimize를 실행한다. `--calib-path`에는 드론 calib 폴더를 넣는다.

```bash
export CUDA_VISIBLE_DEVICES=""

hailomz optimize resnet_v1_18 --hw-arch hailo8l \
  --har /local/shared_with_docker/DRONE/resnet_v1_18.har \
  --calib-path /local/shared_with_docker/calib/drone
```

수업 실습 로그처럼 `Reducing optimization level to 0` 경고가 나와도 정상이다. calib 사진이 권장 수량인 1024장보다 적고 GPU도 없어서 Hailo가 가장 단순한 최적화만 적용한다는 뜻이며, 오류가 아니다. 로그 끝에 `Model Optimization is done`과 `Saved HAR to`가 나오면 성공이다.

optimize는 같은 이름의 HAR 파일을 덮어쓰므로, compile에 넣을 파일을 따로 복사해 둔다. 로그의 `Saved HAR to`에 다른 파일 이름이 찍혔다면 그 파일을 복사한다.

```bash
cp resnet_v1_18.har drone_quantized.har
```


## 7. Compile

```bash
hailomz compile resnet_v1_18 --hw-arch hailo8l \
  --har /local/shared_with_docker/DRONE/drone_quantized.har
```

수업 실습에서는 compile이 1분 남짓 걸렸다. 로그 끝에 `HEF file written to resnet_v1_18.hef`가 나오면 성공이며, 결과 파일을 알아보기 쉬운 이름으로 output 폴더에 복사한다.

```bash
find . -name "*.hef"
mkdir -p /local/shared_with_docker/output
cp resnet_v1_18.hef /local/shared_with_docker/output/drone_resnet18.hef
```


## 8. HEF 확인

```bash
hailortcli parse-hef /local/shared_with_docker/output/drone_resnet18.hef
```

실제로 빌드한 드론 모델은 아래와 같이 나왔다. 수업의 mobilenet_v1은 출력이 `NC(1001)`이었고, 드론 모델은 클래스가 세 개이므로 `NC(3)`이 나온다.

```
Architecture HEF was compiled for: HAILO8L
Network group name: resnet_v1_18, Single Context
    Network name: resnet_v1_18/resnet_v1_18
        VStream infos:
            Input  resnet_v1_18/input_layer1 UINT8, NHWC(224x224x3)
            Output resnet_v1_18/fc1 UINT8, NC(3)
```

이 결과는 라즈베리파이 코드를 짤 때 필요하므로 화면을 캡처해 둔다.


## 9. Windows로 꺼내기

컨테이너에서 나와 Ubuntu에서 공유 폴더로 복사한다.

```bash
exit
mkdir -p /media/sf_NPU_SHARE/output
cp ~/hailo_build/shared_with_docker/output/drone_resnet18.hef /media/sf_NPU_SHARE/output/
ls -lh /media/sf_NPU_SHARE/output/
```

Windows의 `C:\NPU_SHARE\output\drone_resnet18.hef`가 보이면 HEF 빌드가 끝난 것이다.


## 10. 라즈베리파이 코드에 넘길 정보

parse-hef 결과는 라즈베리파이에서 입력을 어떻게 만들어야 하는지 알려 준다. 맥북에서 학습할 때와 입력 형식이 다르므로 아래 세 가지를 지켜야 한다.

1. 입력은 0부터 255까지의 정수(UINT8) RGB 사진을 그대로 넣는다. `resnet_v1_18`의 Model Zoo 설정에 ImageNet 평균과 표준편차로 정규화하는 층이 들어 있어서 HEF가 칩 안에서 정규화를 하므로, 라즈베리파이에서 Normalize를 또 하면 두 번 하게 된다. 이 평균과 표준편차는 b02 학습에 쓴 값과 같다.
2. 크기는 b02 검증용 전처리와 같게 짧은 변을 256으로 줄이고 가운데 224×224를 잘라 낸다.
3. 배열 순서는 NHWC(세로, 가로, 채널)이다. PyTorch와 ONNX에서 쓰던 NCHW(채널이 먼저)와 다르다.

출력은 세 클래스의 logits가 UINT8로 양자화된 값이다. 확률로 쓰려면 실수로 되돌린 뒤 softmax를 적용해야 하며, 클래스 순서는 bird, drone, sky다.


## 11. 오류가 났을 때

| 증상 | 원인 | 해결 |
|---|---|---|
| parse 중 `PermissionError: Permission denied: '....har'` | 작업 폴더 권한이 775라 컨테이너가 쓰지 못함 | 컨테이너에서 `exit` 후 Ubuntu에서 `chmod -R 777 ~/hailo_build/shared_with_docker`를 실행하고 `--resume`으로 다시 들어간다 |
| `Unable to create log file ... (Permission denied)` 경고 | 위와 같은 권한 문제 | 위와 같다 |
| `Can't start a new container, already found one` | 기존 컨테이너가 남아 있음 | `--resume`으로 들어간다 |
| `cp: cannot stat '....har'` | 파일 이름 오타 또는 이전 단계 실패 | `ls`로 실제 파일 이름을 확인한다 |
| `Reducing optimization level to 0` 경고 | calib 사진 1024장 미만, GPU 없음 | 오류가 아니므로 그대로 진행한다 |
| parse 중 `Unable to find end node names` 같은 노드 오류 | ONNX의 입출력 이름이 지정한 값과 다름 | b03 로그의 `ONNX Inputs`와 `ONNX Outputs` 이름을 확인해 옵션 값을 고친다 |
| `hailomz info resnet_v1_18`이 모델을 찾지 못함 | 설치된 Model Zoo에 해당 모델이 없음 | 이 단계에서 멈추고 로그를 저장한다 |
| `/media/sf_NPU_SHARE`에 접근이 안 됨 | Ubuntu 사용자가 vboxsf 그룹에 없음 | 수업 환경 구축 문서의 공유 폴더 설정을 다시 확인한다 |

이 문서의 명령은 2026년 11월 8일에 Hailo-8/8L AI Software Suite 2025-10 컨테이너에서 드론 모델로 처음부터 끝까지 실행해 HEF 생성까지 확인했다. 실제 진행에서 막힌 곳은 3절의 권한 열기를 빠뜨려 작업 폴더에 쓰지 못한 한 번뿐이었다.


## 12. 다른 모델에 다시 쓸 때 바꿀 값

| 항목 | 드론 모델 값 | 바꾸는 기준 |
|---|---|---|
| ONNX 파일 | `b03_resnet18_drone.onnx` | 새로 만든 ONNX 파일 이름 |
| Model Zoo 설정 이름 | `resnet_v1_18` | 같은 구조의 Model Zoo 모델 (MobileNetV2라면 `mobilenet_v2_1.0`처럼 `hailomz info`로 확인) |
| 입력 노드 이름 | `input` | ONNX export 때 `input_names`에 넣은 값 |
| 출력 노드 이름 | `logits` | ONNX export 때 `output_names`에 넣은 값 |
| calib 폴더 | `calib/drone` | 새 모델의 실제 입력과 비슷한 사진 100장을 담은 폴더 |
| 작업 폴더 | `DRONE` | 모델마다 새 폴더를 만든다 |
| 결과 파일 이름 | `drone_resnet18.hef` | 알아보기 쉬운 이름 |
