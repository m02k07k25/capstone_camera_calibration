# Capstone Camera Calibration

캡스톤 프로젝트용 ChArUco 카메라 보정과 스테레오 3D 관절 수집 도구입니다. 카메라 보정, 2D 관절 검출, 3D 재구성 코드는 각각 calibration, pose_estimation.py, pose3d 모듈에 나뉘어 있습니다.

## 입출력 한눈에 보기

| 단계 | 입력 | 출력 |
|---|---|---|
| 보드 생성 | 보드 규격 옵션 | A4 인쇄용 SVG 네 장 |
| 보정 촬영 | 카메라 영상과 ChArUco 보드 | 카메라별 PNG 또는 같은 번호의 이미지 쌍 |
| 단안 보정 | 한 카메라의 보정 이미지 | 카메라 내부 파라미터 NPZ, 오차와 설정 JSON |
| 스테레오 보정 | 양쪽 보정 이미지, 두 단안 결과 | 카메라 사이 파라미터 NPZ와 요약 JSON |
| 2D 이미지 추론 | 이미지, YOLO 포즈 모델 | 관절이 표시된 이미지와 픽셀 좌표 JSON |
| 2D 실시간 추론 | 카메라 두 대, YOLO 포즈 모델 | S 키를 누를 때 2D JSON과 표시 이미지 |
| 3D 실시간 수집 | 카메라 두 대, YOLO 모델, 스테레오 보정 NPZ | 유효한 프레임의 3D 관절 JSON, 선택적 원본 JPEG |
| 저장된 2D에서 3D 생성 | pose_live의 2D JSON, 스테레오 보정 NPZ | 별도 타임스탬프 세션과 3D 관절 JSON |

경로는 프로젝트 루트 기준입니다. 보정 이미지와 결과물은 로컬에서 생성하며 Git에는 포함하지 않습니다. YOLO 모델 파일도 별도로 준비해야 합니다.

## 프로젝트 구조

    capstone_camera_calibration/
    ├─ camera_calibration.py       보정 명령 진입점
    ├─ calibration/                보드, 카메라, 촬영, 단안·스테레오 보정
    ├─ pose_estimation.py          2D 관절 추론
    ├─ stereo_pose.py              3D 수집·재구성 명령 진입점
    ├─ pose3d/                     삼각측량, 관절 매핑, 저장, 미리보기
    ├─ data/charuco_board_a4/      A4 인쇄용 보드 SVG 네 장
    ├─ data/calibration/           보정 촬영 이미지 (로컬 생성)
    ├─ outputs/                    보정 결과와 관절 데이터 (로컬 생성)
    ├─ models/yolo26n-pose.pt      YOLO 포즈 모델 (별도 준비)
    └─ requirements.txt            Python 의존성

## 설치

    python -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt

OpenCV의 ArUco 기능을 사용합니다. opencv-python과 opencv-contrib-python은 같은 cv2 모듈을 제공하므로 한 가상환경에 둘 다 설치하지 마세요.

    .\.venv\Scripts\python.exe -c "import cv2; print(cv2.__version__); print(hasattr(cv2, 'aruco'))"

NVIDIA GPU를 사용하는 경우 의존성 설치 후 CUDA용 PyTorch를 설치합니다. 아래 명령은 이 프로젝트에서 사용한 RTX 3070 환경 예시입니다.

    .\.venv\Scripts\python.exe -m pip install --upgrade --force-reinstall --no-deps torch==2.14.0+cu130 torchvision==0.29.0+cu130 --index-url https://download.pytorch.org/whl/cu130

    .\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

## 1. A4 ChArUco 보드 준비

기본 보드는 가로 10칸, 세로 7칸이며 한 칸은 40 mm, 마커는 28 mm입니다. 전체 보드 크기는 400 × 280 mm이고 내부 코너는 9 × 6개입니다. 각 마커의 ID로 위치와 방향을 구분하므로 체커보드의 좌우·180도 반전 대응 문제를 피할 수 있습니다.

    .\.venv\Scripts\python.exe camera_calibration.py generate-board --output-dir data/charuco_board_a4

data/charuco_board_a4에 A4 가로 SVG 네 장이 생성됩니다. A1/A2는 위쪽 3행, B1/B2는 아래쪽 4행입니다. 각 페이지를 실제 크기 100%로 인쇄하고 페이지 맞춤/축소를 끕니다. 100 mm 확인선을 자로 확인한 뒤 A1 A2 / B1 B2 순서로 이어 붙여 평평하고 단단한 판에 부착합니다.

보드 규격을 바꾸면 생성, 촬영, 단안 보정, 스테레오 보정 명령에 같은 cols, rows, square-size, marker-size, dictionary 값을 사용해야 합니다. 기존 체커보드 이미지는 이 ChArUco 검출기로 사용할 수 없으므로 새 보드로 다시 촬영하세요.

## 2. 카메라 보정 이미지 수집

카메라 인덱스를 확인합니다.

    .\.venv\Scripts\python.exe camera_calibration.py scan --max-index 5 --width 1920 --height 1080

두 카메라가 0번과 1번이면 다음 명령으로 쌍을 촬영합니다.

    .\.venv\Scripts\python.exe camera_calibration.py capture-pair --camera-a 0 --camera-b 1 --backend dshow --cols 10 --rows 7 --square-size 40 --marker-size 28 --dictionary DICT_5X5_100 --width 1920 --height 1080 --preview-width 1400 --preview-height 700

양쪽 영상에 같은 코너 ID가 8개 이상 잡히고 보드가 2초간 안정되면 이미지 한 쌍을 저장합니다. 기본 저장 간격은 4초입니다. 보드를 여러 위치·거리·기울기로 옮기고 저장 순간에는 잠시 멈춥니다. Q 또는 ESC를 눌러 종료합니다.

    data/calibration/camera_0/calibration_000.png
    data/calibration/camera_1/calibration_000.png

같은 번호의 두 파일은 한 쌍입니다. 두 카메라의 프레임은 연속으로 가져오며 하드웨어 트리거로 동기화하지 않습니다. 단안 보정 이미지만 따로 촬영할 때는 capture 명령을 사용합니다. 스테레오 보정에는 양쪽에서 같은 보드 자세를 찍은 쌍이 필요하므로 capture-pair를 권장합니다.

## 3. 단안 카메라 보정

카메라별로 서로 다른 위치·거리·기울기의 이미지 15~20장 이상을 준비하고 각각 보정합니다.

    .\.venv\Scripts\python.exe camera_calibration.py calibrate --input-dir data/calibration/camera_0 --output outputs/camera_0_calibration.npz --corners-dir outputs/camera_0_corners --preview-dir outputs/camera_0_undistorted

    .\.venv\Scripts\python.exe camera_calibration.py calibrate --input-dir data/calibration/camera_1 --output outputs/camera_1_calibration.npz --corners-dir outputs/camera_1_corners --preview-dir outputs/camera_1_undistorted

각 NPZ에는 카메라 행렬, 렌즈 왜곡 계수, 이미지 해상도와 보드 설정이 저장됩니다. 같은 이름의 JSON에는 재투영 오차와 사용 이미지 목록이 기록됩니다. 코너 검출 이미지와 왜곡 보정 미리보기는 지정한 폴더에 저장됩니다. square-size 단위는 이동 벡터에도 사용되므로 이 프로젝트의 기본 단위인 mm를 유지하세요.

## 4. 스테레오 보정

같은 번호의 이미지 쌍과 두 단안 보정 결과로 카메라 사이의 회전·이동 및 영상 정렬 파라미터를 계산합니다.

    .\.venv\Scripts\python.exe camera_calibration.py stereo-calibrate --input-a data/calibration/camera_0 --input-b data/calibration/camera_1 --calibration-a outputs/camera_0_calibration.npz --calibration-b outputs/camera_1_calibration.npz --output outputs/stereo_calibration.npz

출력 NPZ에는 R/T, E/F, rectification, Q 행렬이 저장되고 같은 이름의 JSON에는 오차와 이미지 쌍별 공통 코너 수가 기록됩니다. R/T의 거리 단위는 square-size 입력 단위인 mm입니다. 스테레오 RMS가 기본 기준 5 px보다 크면 결과를 저장하지 않습니다. 보드의 움직임, 흐림, 인쇄 크기, 이미지 해상도와 보드 규격을 확인하고 다시 촬영하세요. ChArUco ID는 코너 대응 순서 문제를 줄여 주지만 촬영 품질 문제까지 보정하지는 않습니다.

## 5. 왜곡 보정과 2D 관절 검출

한 장의 이미지를 보정하려면 다음처럼 실행합니다.

    .\.venv\Scripts\python.exe camera_calibration.py undistort --calibration outputs/camera_0_calibration.npz --input input.jpg --output outputs/undistorted.jpg

2D 관절 검출에는 models/yolo26n-pose.pt 모델이 필요합니다. 이미지 추론은 관절 표시 이미지와 같은 파일명의 JSON을 출력합니다. 관절 x/y는 이미지 픽셀이고 confidence는 모델 신뢰도입니다.

    .\.venv\Scripts\python.exe pose_estimation.py image --model models/yolo26n-pose.pt --input input.jpg --output outputs/pose_test.png --imgsz 640 --conf 0.35

두 카메라 실시간 2D 추론은 다음과 같습니다. S를 누르면 outputs/pose_live에 pose_NNN.json, pose_NNN_camera_a.png, pose_NNN_camera_b.png가 저장됩니다. JSON에는 저장 시각(timestamp, Unix 초), 두 영상 크기, COCO-17 2D 키포인트의 픽셀 좌표와 confidence가 들어 있습니다.

    .\.venv\Scripts\python.exe pose_estimation.py live --model models/yolo26n-pose.pt --camera-a 0 --camera-b 1 --backend dshow --width 1920 --height 1080 --preview-width 1400 --preview-height 700 --imgsz 640 --conf 0.35 --device 0

Q 또는 ESC로 종료합니다. 기본 출력 폴더는 outputs/pose_live입니다.

## 6. 실시간 3D 관절 저장

스테레오 보정 결과와 카메라 두 대로 실행합니다.

    .\.venv\Scripts\python.exe stereo_pose.py live --model models/yolo26n-pose.pt --stereo-calibration outputs/stereo_calibration.npz --camera-a 0 --camera-b 1 --backend dshow --width 1920 --height 1080 --preview-width 1600 --preview-height 700 --imgsz 640 --conf 0.35 --device 0

저장은 자동입니다. 원시 삼각측량 결과에서 유효한 COCO 관절이 기본 6개 이상인 프레임을 처리 직후 JSON으로 씁니다. 기본 --save-interval 0은 유효한 모든 프레임을 저장합니다. 예를 들어 --save-interval 0.1은 초당 최대 약 10개로 제한합니다. Q 또는 ESC로 종료합니다. 기본적으로 JSON만 저장합니다. 카메라 원본 JPEG도 보관하려면 명령에 --save-images 옵션을 추가하세요.

한 사람을 각 카메라에서 독립적으로 검출해 대응시키므로 수집 중에는 한 사람만 화면에 두는 것을 권장합니다. 카메라 해상도는 스테레오 보정 당시 해상도와 같아야 합니다.

각 실행은 outputs/pose_3d 아래에 새 UTC 타임스탬프 폴더를 만듭니다.

    outputs/pose_3d/20261001T123456.123456789Z/
    ├─ session.json
    ├─ frames/
    │  └─ frame_000000_20261001T123456.123456789Z.json
    └─ images/                         save-images를 준 경우에만 생성
       ├─ camera_a/frame_000000.jpg
       └─ camera_b/frame_000000.jpg

session.json에는 모델, 카메라, 보정 파일, 저장 조건 등 세션 설정이 기록됩니다. 각 frames JSON에는 해당 프레임의 카메라별 2D 검출, 원시 3D COCO-17 관절(coco17_3d), 필터링한 관절(coco17_3d_filtered), SMPL 이름 순서의 24개 XYZ 위치(smpl_body_24와 smpl_body_24_filtered)가 담깁니다. JSON은 매 프레임 디스크에 기록되므로 수집 중에도 파일이 누적됩니다. 학습에는 원시 필드를 사용하고, 필터링 필드에는 흔들림 완화나 짧은 결측 구간 유지가 반영될 수 있습니다.

좌표 단위는 mm입니다. 원점은 rectified camera A의 광학 중심이며 X는 영상 오른쪽, Y는 아래쪽, Z는 카메라 앞쪽입니다. 각 3D COCO 관절에는 xyz_mm, valid, confidence, reprojection_error_px가 들어갑니다. SMPL BODY 24개는 smpl_body_24 배열에 이름 순서대로 들어가며 각 항목에 name, xyz_mm, source가 기록됩니다. 관측하지 않은 값은 null입니다. source가 direct가 아닌 관절은 COCO 관절에서 계산한 중간점 또는 proxy 값일 수 있습니다.

프레임에는 다음 시간값이 기록됩니다.

- timestamp_utc: UTC 시간 문자열
- timestamp_unix_ns: Unix 기준 나노초
- timestamp_monotonic_ns: 동일 컴퓨터의 단조 시계 기준 나노초

실시간 수집 시각은 양쪽 카메라 프레임을 순서대로 가져온 직후 호스트 컴퓨터에서 기록합니다. 카메라는 하드웨어 동기화되지 않습니다. 같은 컴퓨터에서 실행한 IMU 기록과 맞출 때는 monotonic 시각을 사용하고, 다른 장치의 IMU는 시계 오프셋을 별도로 맞춰야 합니다. 저장 빈도는 고정 25 Hz가 아니라 추론 속도에 따릅니다. IMU 샘플링 속도에 맞춰 보간하거나 리샘플링하세요.

## 7. 저장된 2D 관절에서 3D 생성

pose_estimation.py live로 저장한 pose_*.json도 나중에 삼각측량할 수 있습니다.

    .\.venv\Scripts\python.exe stereo_pose.py reconstruct --input-dir outputs/pose_live --output-dir outputs/pose_3d --stereo-calibration outputs/stereo_calibration.npz --min-keypoint-conf 0.35

입력 JSON의 camera_a/camera_b 2D 관절을 사용해 새 타임스탬프 세션에 3D JSON을 만듭니다. 실시간 3D 수집과 달리 원본 카메라 프레임은 포함하지 않습니다. 2D JSON에 monotonic 시간이 없으므로 재구성 결과의 timestamp_monotonic_ns는 null이며, 저장 당시의 Unix 시각을 사용합니다.

## 학습 데이터로 사용할 때

이 프로젝트의 3D 관절은 스테레오 영상에서 추정한 pseudo-label입니다. 모션 캡처 ground truth가 아닙니다. smpl_body_24는 COCO-17 검출 결과를 24개 SMPL 관절 이름에 맞춰 매핑한 XYZ 위치이며, SMPL 피팅 결과나 메시, SMPL 포즈 파라미터가 아닙니다. 일부 관절은 중간점 또는 proxy로 채워질 수 있으므로 source와 valid 필드를 확인하세요.

IMUPoser는 Vicon 모션 캡처를 기준 포즈로 사용하고 그 데이터에 SMPL을 맞춥니다 ([논문](https://arxiv.org/abs/2304.12518)). 따라서 이 프로젝트의 출력을 정답 데이터로 채택하기 전 재투영 오차와 누락 관절을 검토하고, 모션 캡처나 수동 검수 결과와 비교해 정확도를 확인하세요.
