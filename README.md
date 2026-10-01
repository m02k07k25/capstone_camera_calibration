# Capstone Camera Calibration

ChArUco 카메라 보정 → COCO-17 3D 관찰값 수집 → 오프라인 시간 보정 → 선택적 SMPL 피팅 도구입니다.

**COCO-17 좌표를 24개 관절 이름에 끼워 넣은 값은 SMPL이 아닙니다.** 새 JSON은 `schema_version=2`이며, 피팅 전에는 `smpl.status="not_fitted"`입니다. 실제 SMPL 모델을 이용하는 `pose_dataset.py fit-smpl`만 SMPL 파라미터를 출력합니다. 이 결과 역시 모션캡처 ground truth가 아닌 추정 pseudo-label입니다.

## 설치

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python -c "import cv2; print(cv2.__version__, hasattr(cv2, 'aruco'))"
```

`opencv-python`과 `opencv-contrib-python`을 같은 환경에 중복 설치하지 마세요. GPU를 사용하려면 설치된 PyTorch의 CUDA 지원 여부를 확인하세요. YOLO 모델 `models/yolo26n-pose.pt`는 별도로 준비합니다. Linux에서는 카메라 명령에 `--backend auto`를 사용하세요.

## 1. A4 한 장 ChArUco 보드

기본값: **7×5칸, 한 칸 35 mm, 마커 24.5 mm, DICT_5X5_100**. 전체 패턴 245×175 mm, 내부 코너 24개입니다.

```bash
python camera_calibration.py generate-board --output-dir data/charuco_board_a4
```

생성된 `charuco_board_A4.svg`를 **A4 가로 / 실제 크기 100% / 페이지 맞춤 해제**로 출력합니다. 100 mm 확인선을 실측하고 평평한 판에 부착하세요. 보드 생성·검출에 같은 규격을 사용해야 합니다. 예전 10×7 보드나 체커보드 이미지를 새 기본값으로 처리하지 마세요.

## 2. 카메라 보정

```bash
python camera_calibration.py scan --max-index 5 --width 1920 --height 1080
python camera_calibration.py capture-pair --camera-a 0 --camera-b 1 --backend dshow --width 1920 --height 1080
python camera_calibration.py calibrate --input-dir data/calibration/camera_0 --output outputs/camera_0_calibration.npz --corners-dir outputs/camera_0_corners --preview-dir outputs/camera_0_undistorted
python camera_calibration.py calibrate --input-dir data/calibration/camera_1 --output outputs/camera_1_calibration.npz --corners-dir outputs/camera_1_corners --preview-dir outputs/camera_1_undistorted
python camera_calibration.py stereo-calibrate --input-a data/calibration/camera_0 --input-b data/calibration/camera_1 --calibration-a outputs/camera_0_calibration.npz --calibration-b outputs/camera_1_calibration.npz --output outputs/stereo_calibration.npz
```

보드를 여러 위치·거리·기울기에서 촬영하고 저장 순간에는 정지하세요. `capture-pair`는 공통 코너 8개 이상과 2초 안정성을 확인합니다. 15~20장 이상의 다양한 영상을 준비하되, 장수만으로 품질이 보장되지는 않습니다. 카메라 하나만 촬영하는 `capture`, 특정 쌍을 제외하는 `stereo-calibrate --exclude-pairs`도 유지됩니다.

단안 NPZ에는 K/D/해상도/보드 규격, 스테레오 NPZ에는 R/T/E/F/rectification/projection/Q가 저장됩니다. 거리 단위는 기본 mm입니다. 스테레오 기본 `--max-rms 5`는 오류 차단용 설정이지 3D 정확도 인증 기준이 아닙니다. 실제 거리·재투영·별도 검증 영상을 확인하세요. 단안의 `per_view_reprojection_error`는 수정된 RMS 식을 사용하며, 그 단순 평균은 전체 점을 가중한 global RMS와 다릅니다.

## 3. 2D 추론과 기존 이미지 보정

```bash
python camera_calibration.py undistort --calibration outputs/camera_0_calibration.npz --input input.jpg --output outputs/undistorted.jpg
python pose_estimation.py image --model models/yolo26n-pose.pt --input input.jpg --output outputs/pose_test.png --imgsz 640 --conf 0.35
python pose_estimation.py live --camera-a 0 --camera-b 1 --backend dshow --width 1920 --height 1080 --device 0
```

2D live에서 S를 누르면 `outputs/pose_live`에 2D JSON과 표시 이미지가 저장됩니다. Q/ESC로 종료합니다. 보정된 이미지에 원래 K/D를 다시 적용하지 않도록 좌표계에 주의하세요.

## 4. 프레임별 3D 원본 수집

```bash
python stereo_pose.py live --stereo-calibration outputs/stereo_calibration.npz --camera-a 0 --camera-b 1 --backend dshow --width 1920 --height 1080 --device 0 --save-images --save-interval 0 --save-min-valid 0
```

이제 기본 `save-min-valid=0`이므로 저품질·결측 프레임도 mask와 함께 보존합니다. `--save-images`를 주면 획득한 카메라 JPEG도 저장합니다. 원본 JSON/JPEG를 남겨 두고 파생 데이터를 별도 파일로 만드세요. JPEG는 손실 압축이며 보존율과 인코딩/디스크 부하를 확인해야 합니다.

**현재 live는 카메라 획득과 YOLO 추론이 같은 루프입니다. 이 옵션이 카메라의 모든 30 fps 프레임을 무손실로 녹화한다는 뜻은 아닙니다.** 처리율이 느리면 획득하지 못한 프레임은 나중에 복구할 수 없습니다. 모든 카메라 프레임 보존이 필요한 수집에는 독립 녹화/획득 파이프라인이 추가로 필요합니다. 카메라 간 하드웨어 동기화도 구현되어 있지 않습니다.

각 실행은 `outputs/pose_3d/<session>/session.json`, `frames/*.json`, 선택적 `images/camera_a|camera_b/*.jpg`를 만듭니다. 한 사람만 촬영하고 카메라 해상도는 보정 때와 동일하게 유지하세요.

새 프레임 스키마:

| 필드 | 의미 |
|---|---|
| `coco17_3d` | 17개 XYZ 관찰값, mm, validity/confidence/reprojection error |
| `coco17_3d_filtered` | 기존 실시간 표시용 필터·hold 결과. 독립 관측/학습 정답 아님 |
| `smpl` | 피팅 전 `status=not_fitted`, 파라미터 null |
| `timestamp_*` | 호스트의 Unix/monotonic 시간. 카메라 노출 시각 보장 아님 |

`smpl_body_24`와 `smpl_body_24_filtered`는 새 JSON에서 제거했습니다. 화면의 Body24 PROXY는 표시용 근사일 뿐이며 실제 SMPL 피팅과 별개입니다. 기존 v1 데이터는 **원본 `coco17_3d`만** 읽어 새 오프라인 파이프라인에 넣을 수 있습니다. 2D JSON에서의 기존 재구성 명령도 유지됩니다.

```bash
python stereo_pose.py reconstruct --input-dir outputs/pose_live --output-dir outputs/pose_3d --stereo-calibration outputs/stereo_calibration.npz
```

## 5. 오프라인 보정 → 학습 윈도우 → SMPL

**먼저 사람/세션 단위로 train/val/test를 정하세요.** 같은 촬영의 겹친 윈도우를 나중에 무작위 분할하면 데이터 누수가 발생합니다. 아래 명령은 하나의 전체 세션을 처리합니다. 경로의 `SESSION_ID`는 실제 촬영 폴더로 바꾸세요.

```bash
python pose_dataset.py smooth --input outputs/pose_3d/SESSION_ID --output outputs/session_smoothed.npz --max-gap-seconds 0.2
python pose_dataset.py windows --input outputs/session_smoothed.npz --output outputs/session_train.npz --fps 30 --window-seconds 1.0 --stride-seconds 0.2 --subject-id P01 --split train
```

`smooth`는 실제 timestamp 간격을 이용하는 **Kalman + backward RTS smoother**입니다. `--max-gap-seconds 0.2`는 메울 수 있는 관측 공백의 한계이며, 6프레임만 쓰는 smoothing window나 5 Hz 출력 주기가 아닙니다. 전체 연결 구간의 과거·미래를 사용하되 긴 결측을 건너뛰거나 끝을 외삽하지 않습니다. `observed_points_mm`, `observed_valid`, `imputed_mask`를 따로 보존합니다. `--measurement-std-mm`와 `--acceleration-std-mm-s2`는 검증/조정해야 하는 모델 가정입니다.

`windows`는 timestamp 기준으로 30 Hz 격자를 만든 뒤 **원본 시퀀스 + [start, stop) 인덱스**를 저장합니다. 겹친 배열을 수십 번 복제하지 않습니다. 보간은 독립 관측을 늘리지 않습니다. 윈도우별 `window_valid_fraction`을 확인하고 긴 결측/저품질 구간은 학습에서 제외하세요.

1초 윈도우/0.2초 stride는 0.8초 overlap입니다. 0.5초 overlap은 1초 윈도우에서 `--stride-seconds 0.5`입니다. 0.2초 **길이**의 윈도우에 0.5초 overlap은 불가능합니다.

실제 SMPL을 피팅하려면 선택적 라이브러리와 본인이 내려받은 정식 SMPL 모델이 필요합니다.

```bash
python -m pip install -r requirements-smpl.txt
python pose_dataset.py fit-smpl --input outputs/session_train.npz --output outputs/session_smpl.npz --model-path models/smpl/SMPL_NEUTRAL.pkl --gender neutral --device cuda:0
```

모델 파일은 이 저장소에 포함되지 않습니다. 파일이 없으면 실패하며 COCO를 가짜 SMPL로 대체하지 않습니다. 입력의 window index와 subject/split metadata는 SMPL 결과에도 보존됩니다. 자세한 출력과 한계는 [SMPL 및 시계열 데이터 계약](docs/pose_data_contract.md)을 참고하세요.

## 테스트와 검증 범위

```bash
python -m pip install pytest
python -m pytest -q
```

테스트는 스키마, 결측/시간 처리, overlap 인덱스, 합성 신호 RTS, 회전 수학 및 toy body-model 인터페이스를 확인합니다. 실제 카메라·실제 인체 영상·정식 SMPL 가중치에 대한 정확도 검증을 대신하지 않습니다.
