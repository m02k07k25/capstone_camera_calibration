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
├─ requirements.txt
├─ data/
│  └─ calibration/       # 체스보드 촬영 이미지
├─ outputs/              # 캘리브레이션 결과와 보정 이미지
└─ .gitignore
```

## 설치

Python 3.9 이상을 권장합니다.

```bash
python -m pip install -r requirements.txt
```

## 1. 체스보드 이미지 촬영

기본 설정은 내부 코너 `9 x 6`, 한 칸 크기 `25.0`입니다. 여기서 코너 수는 체스보드 칸 수가 아니라 내부 교차점 수입니다.

```bash
python camera_calibration.py capture
```

체스보드를 여러 각도와 위치에서 촬영합니다. 화면에서 코너가 초록색으로 검출될 때 스페이스 또는 `S`를 누르면 `data/calibration`에 저장됩니다. `Q` 또는 `ESC`로 종료합니다.

체스보드 규격이 다르면 실제 내부 코너 수와 칸 크기를 지정합니다.

```bash
python camera_calibration.py capture --cols 9 --rows 6 --square-size 25
```

## 2. 캘리브레이션 실행

```bash
python camera_calibration.py calibrate --input-dir data/calibration --cols 9 --rows 6 --square-size 25 --output outputs/calibration.npz --preview-dir outputs/undistorted
```

결과:

- `outputs/calibration.npz`: 프로그램에서 읽는 카메라 행렬과 왜곡 계수
- `outputs/calibration.json`: 사람이 읽을 수 있는 요약 정보
- `outputs/corners/`: 검출된 코너 확인 이미지
- `outputs/undistorted/`: 선택한 경우 생성되는 왜곡 보정 이미지

`square-size`의 단위는 결과의 이동 벡터 단위가 되므로, 실제 측정 단위를 일관되게 사용하면 됩니다. 캘리브레이션 이미지들은 같은 해상도로 촬영해야 합니다.

## 3. 단일 이미지 보정

```bash
python camera_calibration.py undistort --calibration outputs/calibration.npz --input input.jpg --output outputs/undistorted.jpg
```

폴더 전체를 보정할 수도 있습니다.

```bash
python camera_calibration.py undistort --calibration outputs/calibration.npz --input data/calibration --output outputs/undistorted
```

재투영 오차는 작을수록 일반적으로 좋지만, 이미지 품질과 체스보드 촬영 각도에 따라 달라집니다. 결과를 사용할 때는 `calibration.json`의 오차와 `outputs/corners` 이미지를 함께 확인하세요.
