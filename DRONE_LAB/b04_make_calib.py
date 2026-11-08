from pathlib import Path
import random
import shutil


# ============================================================
# 1. 경로 설정
#
# 원본 : data/drone_cls/{bird,drone,sky}
# 결과 : calib/drone/calib_001_drone.jpg ...
#        (Hailo optimize --calib-path 로 넘길 폴더)
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent

SOURCE_DIR = SCRIPT_DIR / "data" / "drone_cls"

CALIB_DIR = SCRIPT_DIR / "calib" / "drone"


# ============================================================
# 2. 설정
#
# 수업 Step 9과 동일하게 100장 사용
# ============================================================
SEED = 42

CALIB_COUNTS = {
    "drone": 34,
    "bird": 33,
    "sky": 33,
}


random.seed(SEED)

CALIB_DIR.mkdir(parents=True, exist_ok=True)

for old_file in CALIB_DIR.glob("*.jpg"):
    old_file.unlink()


# ============================================================
# 3. 클래스별 무작위 선택 후 복사
# ============================================================
index = 1

for class_name, count in CALIB_COUNTS.items():

    files = sorted((SOURCE_DIR / class_name).glob("*.jpg"))

    for source_file in random.sample(files, count):

        target_file = CALIB_DIR / f"calib_{index:03d}_{class_name}.jpg"

        shutil.copy2(source_file, target_file)

        index += 1

    print(f"[INFO] {class_name:5s} : {count}")

print("[OK] Calibration images :", len(list(CALIB_DIR.glob("*.jpg"))))
print("[OK] Calibration dir    :", CALIB_DIR)
