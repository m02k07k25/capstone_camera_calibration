# IMU Rehab Dataset

AMASS/SMPL 동작과 합성 어깨 재활 동작에서 가상 IMU 시계열 데이터를 생성하고, 검증 및 학습용 구간으로 정리하는 Python 스크립트 모음입니다. 스크립트는 저장소 루트에서 실행하며, 데이터와 결과 파일은 저장소에 포함하지 않습니다.

## 처리 흐름

### AMASS 동작

1. `ACCAD/` 아래 원본 동작 파일을 검색하고 상체 동작 후보를 만듭니다.
2. 후보 동작을 SMPL 모델로 처리하고 50 Hz 가상 IMU 데이터를 생성합니다.
3. 결과를 검증·요약하고, 필요하면 학습용 윈도우로 분할합니다.

```powershell
python select_amass_upper_body.py
python build_amass_dataset_v1.py --limit 1
python validate_amass_dataset_v1.py
python summarize_amass_dataset_v1.py
python create_training_windows.py
```

`--limit 1`은 소량 시험 실행 옵션입니다. 전체 후보를 처리하려면 해당 옵션을 생략하세요.

### 합성 재활 동작

어깨 굴곡·외전·견갑면 거상 동작과 몸통 보상 동작을 생성한 뒤, SMPL 기반 가상 IMU 데이터를 만들고 AMASS 결과와 통합합니다.

```powershell
python create_shoulder_rehab_dataset.py
python build_rehab_dataset.py
python combine_final_virtual_dataset.py
python view_amass_dataset_motion.py --type rehab --index 1
```

학습 윈도우 스크립트(`create_training_windows.py`)는 현재 `outputs/amass_dataset_v1/`을 입력으로 사용합니다.

## 환경 준비

Python 3.11 환경을 기준으로 작성되었습니다. 기본 패키지를 설치하고, 사용하는 시스템에 맞는 PyTorch 설치 명령은 [PyTorch 설치 안내](https://pytorch.org/get-started/locally/)를 따르세요.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy scipy matplotlib smplx
```

PyTorch와 `smplx` 설치가 끝나면, 필요한 SMPL 모델 파일을 `models/` 아래에 별도로 준비하세요. `python test_smpl.py`로 모델 로딩을 확인할 수 있습니다.

## 입력 데이터 및 라이선스

- AMASS 호환 동작 파일(`*_poses.npz`)을 `ACCAD/` 아래에 준비해야 합니다.
- SMPL 모델 파일은 저장소에 포함되어 있지 않습니다.
- 원본 동작 데이터와 SMPL 모델은 각 제공자의 라이선스에 따라 별도로 취득하고 사용해야 합니다.
- 이 저장소의 `.gitignore`는 원본 데이터, 모델, 생성된 `.npz` 및 `outputs/`를 제외합니다. 특히 포함된 ACCAD 라이선스는 데이터셋 재배포를 허용하지 않으므로 원본 또는 파생 데이터 파일을 공개 저장소에 올리지 마세요.

## 주요 스크립트

| 스크립트 | 역할 |
| --- | --- |
| `select_amass_upper_body.py` | 상체 중심 AMASS 동작 후보 선별 |
| `build_amass_dataset_v1.py` | 원본 동작을 처리해 50 Hz 가상 IMU 데이터 생성 |
| `create_shoulder_rehab_dataset.py` | 합성 어깨 재활 동작 생성 |
| `build_rehab_dataset.py` | 재활 동작을 가상 IMU 데이터로 변환 |
| `combine_final_virtual_dataset.py` | AMASS 및 재활 결과 통합 |
| `create_training_windows.py` | IMU 및 자세 데이터를 학습 구간으로 분할 |
| `validate_*.py`, `summarize_amass_dataset_v1.py` | 결과 검증 및 요약 |
| `view_amass_dataset_motion.py` | 통합 데이터의 3D 동작 재생 |

## 출력 위치

- `outputs/amass_dataset_v1/`: AMASS 기반 가상 IMU 데이터
- `outputs/rehab_dataset/`: 합성 재활 동작 기반 가상 IMU 데이터
- `outputs/final_virtual_dataset/`: 통합 데이터와 인덱스
- `outputs/training_windows_v1/`: 학습·검증·시험 구간

이 경로의 산출물은 재생성할 수 있으며 버전 관리 대상이 아닙니다.