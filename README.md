# Capstone Camera Calibration

캡스톤 프로젝트에서 사용할 수 있는 체스보드 기반 카메라 캘리브레이션 도구입니다.
웹캠으로 체스보드 이미지를 촬영하거나 기존 이미지를 사용해 다음 작업을 수행합니다.

- 카메라 행렬 `K` 계산
- 렌즈 왜곡 계수 계산
- RMS 및 평균 재투영 오차 계산
- 코너 검출 확인 이미지 저장
- 왜곡 보정 이미지 생성

## 프로젝트 구조

```text
capstone_camera_calibration/
├─ camera_calibration.py
├─ pose_estimation.py
├─ requirements.txt
├─ data/
│  └─ calibration/       # 체스보드 촬영 이미지
├─ outputs/              # 캘리브레이션 결과와 보정 이미지
├─ models/
│  └─ yolo26n-pose.pt  # 사전학습 HPE 가중치
└─ .gitignore
```

## 설치

프로젝트 전용 가상환경을 만들고 의존성을 설치합니다.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

NVIDIA GPU를 사용하는 경우에는 기본 의존성 설치 후 CUDA용 PyTorch를 별도로 설치합니다. 현재 RTX 3070 환경에서는 다음 명령을 사용합니다.

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade --force-reinstall --no-deps torch==2.14.0+cu130 torchvision==0.29.0+cu130 --index-url https://download.pytorch.org/whl/cu130
```

확인 명령에서 `cuda_available=True`와 `NVIDIA GeForce RTX 3070`이 표시되어야 합니다.

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 체커보드 규격

현재 사용하는 보드가 가로 10칸 × 세로 7칸이고 한 칸이 25 mm라면, OpenCV에 입력하는 내부 코너 수는 가로 9 × 세로 6입니다.

```text
실제 칸 수:       10 × 7
내부 교차점 수:    9 × 6  ← OpenCV --cols/--rows
한 칸 크기:       25.0 mm
```

보드에 표시된 숫자가 “칸 수”가 아니라 “내부 교차점 수”라면 그때만 `--cols 10 --rows 7`로 바꿉니다.

## 1. 연결된 카메라 확인

```powershell
.\.venv\Scripts\python.exe camera_calibration.py scan --max-index 5 --width 1920 --height 1080
```

`[0] 사용 가능`, `[1] 사용 가능`처럼 표시되는 번호를 촬영 명령의 카메라 번호로 사용합니다. Windows에서 카메라 앱이나 다른 프로그램이 장치를 점유하고 있으면 먼저 종료합니다.

## 2. 두 카메라로 체커보드 촬영

두 카메라가 각각 0번과 1번으로 잡히는 경우:

```powershell
.\.venv\Scripts\python.exe camera_calibration.py capture-pair --camera-a 0 --camera-b 1 --backend dshow --cols 9 --rows 6 --width 1920 --height 1080 --preview-width 1400 --preview-height 700 --interval 0.5
```

미리보기에서 양쪽 모두 `CORNER OK`가 된 상태로 체커보드의 위치와 각도를 바꾸며 스페이스 또는 `S`를 누릅니다. 두 장의 이미지는 같은 번호로 다음 위치에 저장됩니다.

이 모드는 두 카메라를 먼저 프레임 획득한 뒤 디코딩해 같은 체커보드 자세의 이미지 쌍을 만들도록 합니다. USB 카메라의 하드웨어 트리거 동기화는 아니므로, 저장을 누를 때 체커보드를 잠시 멈추세요.

```text
data/calibration/camera_0/
data/calibration/camera_1/
```

`Q` 또는 `ESC`로 종료합니다. 해상도를 카메라가 지원하지 않으면 실제 지원 해상도로 자동 적용됩니다. 모든 촬영 이미지는 카메라별로 같은 해상도여야 합니다.

기존 촬영 파일이 있으면 다음 번호부터 자동으로 이어서 저장하므로, 추가 촬영을 해도 기존 이미지가 덮어써지지 않습니다.

두 번째 카메라가 아직 인덱스 1로 열리지 않거나 카메라별로 따로 촬영해야 하면 다음처럼 실행합니다.

```powershell
.\.venv\Scripts\python.exe camera_calibration.py capture --camera 0 --output-dir data/calibration/camera_0 --cols 9 --rows 6 --width 1920 --height 1080

.\.venv\Scripts\python.exe camera_calibration.py capture --camera 1 --output-dir data/calibration/camera_1 --cols 9 --rows 6 --width 1920 --height 1080
```

## 3. 카메라별 캘리브레이션

각 카메라에서 최소 15~20장 이상, 서로 다른 위치·거리·기울기로 촬영한 뒤 각각 실행합니다.

```powershell
.\.venv\Scripts\python.exe camera_calibration.py calibrate --input-dir data/calibration/camera_0 --cols 9 --rows 6 --square-size 25 --output outputs/camera_0_calibration.npz --corners-dir outputs/camera_0_corners --preview-dir outputs/camera_0_undistorted

.\.venv\Scripts\python.exe camera_calibration.py calibrate --input-dir data/calibration/camera_1 --cols 9 --rows 6 --square-size 25 --output outputs/camera_1_calibration.npz --corners-dir outputs/camera_1_corners --preview-dir outputs/camera_1_undistorted
```

결과는 카메라별 `.npz`와 `.json`, 검출 코너 이미지, 왜곡 보정 미리보기로 저장됩니다.

`square-size`의 단위는 이동 벡터의 단위가 되므로 실제 측정 단위인 mm를 계속 사용하면 됩니다.

## 4. 스테레오 캘리브레이션

두 카메라의 같은 번호 이미지 쌍을 사용해 카메라 사이의 회전·이동과 영상 정렬 파라미터를 계산합니다. 3D 관절 좌표를 만들기 전에 실행해야 합니다.

```powershell
.\.venv\Scripts\python.exe camera_calibration.py stereo-calibrate --input-a data/calibration/camera_0 --input-b data/calibration/camera_1 --calibration-a outputs/camera_0_calibration.npz --calibration-b outputs/camera_1_calibration.npz --cols 9 --rows 6 --square-size 25 --output outputs/stereo_calibration.npz
```

두 카메라가 서로 다른 순간의 보드를 저장한 쌍은 다음처럼 제외할 수 있습니다. 여러 파일은 쉼표로 구분합니다.

```powershell
.\.venv\Scripts\python.exe camera_calibration.py stereo-calibrate --input-a data/calibration/camera_0 --input-b data/calibration/camera_1 --calibration-a outputs/camera_0_calibration.npz --calibration-b outputs/camera_1_calibration.npz --cols 9 --rows 6 --square-size 25 --exclude-pairs calibration_011.png,calibration_019.png --output outputs/stereo_calibration.npz
```

스테레오 rectification의 기본 alpha는 -1이며, 왜곡이 큰 카메라에서도 과도한 확대를 피하도록 설정되어 있습니다.

결과:

- `outputs/stereo_calibration.npz`: 두 카메라의 R/T, E/F, rectification, Q 행렬
- `outputs/stereo_calibration.json`: RMS 오차와 카메라 사이 거리 요약

스테레오 RMS 오차가 기본 기준인 5 px보다 크면 결과 파일을 저장하지 않습니다. 이 경우 체커보드를 더 넓은 위치·거리·기울기로 움직이고, 저장 순간에는 잠시 멈춰서 다시 촬영합니다.

## 5. 단일 이미지 보정

```powershell
.\.venv\Scripts\python.exe camera_calibration.py undistort --calibration outputs/camera_0_calibration.npz --input input.jpg --output outputs/undistorted.jpg
```

재투영 오차는 작을수록 일반적으로 좋습니다. 캘리브레이션 후에는 `outputs/corners`에 저장된 코너 검출 결과와 `.json`의 오차를 함께 확인하세요.

## 6. HPE 2D 관절 검출

Ultralytics YOLO26 nano pose 모델을 사용합니다. 모델은 17개 COCO 관절의 2D 좌표와 confidence를 출력합니다. 이 단계는 2D 검출이며, 3D 관절 좌표에는 정상적인 스테레오 캘리브레이션이 추가로 필요합니다.

단일 이미지 테스트:

```powershell
.\.venv\Scripts\python.exe pose_estimation.py image --model models/yolo26n-pose.pt --input data/calibration/camera_0/calibration_000.png --output outputs/pose_test.png --imgsz 640 --conf 0.35
```

두 카메라 실시간 HPE:

```powershell
.\.venv\Scripts\python.exe pose_estimation.py live --model models/yolo26n-pose.pt --camera-a 0 --camera-b 1 --backend dshow --width 1920 --height 1080 --preview-width 1400 --preview-height 700 --imgsz 640 --conf 0.35 --device 0
```

화면에서 양쪽 사람 관절이 검출될 때 `S`를 누르면 `outputs/pose_live`에 두 카메라의 2D 관절 JSON과 표시 이미지가 저장됩니다. `Q` 또는 `ESC`로 종료합니다. `--device auto`를 사용하면 CUDA가 있으면 GPU, 없으면 CPU를 자동 선택하고 GPU에서는 FP16을 사용합니다.
