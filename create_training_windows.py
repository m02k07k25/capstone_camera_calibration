from pathlib import Path

import numpy as np


# ============================================================
# 기본 설정
# ============================================================

SOURCE_DIRECTORY = Path(
    "outputs/amass_dataset_v1"
)

OUTPUT_DIRECTORY = Path(
    "outputs/training_windows_v1"
)

FRAME_RATE = 50

# 한 학습 구간의 길이: 2초
WINDOW_SIZE = 100

# 다음 구간까지 이동: 1초
WINDOW_STRIDE = 50

# 미분과 필터로 인한 처음·끝 경계 오차 제거
BOUNDARY_TRIM = 5

# 비정상적으로 큰 IMU 값이 포함된 구간 제외 기준
MAX_ACCELERATION = 80.0
MAX_GYROSCOPE = 15.0

# 데이터 분할 결과를 동일하게 유지
RANDOM_SEED = 42


# ============================================================
# 출력 폴더 생성 및 파일 탐색
# ============================================================

OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)

dataset_files = sorted(
    SOURCE_DIRECTORY.glob("amass_*_imu.npz")
)

if not dataset_files:
    raise FileNotFoundError(
        "AMASS 데이터셋 파일이 없습니다."
    )


# ============================================================
# 원본 AMASS 파일 단위 데이터 분리
# ============================================================

# 같은 원본 동작에서 생성된 비슷한 구간이
# train과 test에 동시에 들어가는 것을 방지한다.
random_generator = np.random.default_rng(
    RANDOM_SEED
)

shuffled_indices = random_generator.permutation(
    len(dataset_files)
)

train_end = int(len(dataset_files) * 0.70)
validation_end = int(len(dataset_files) * 0.85)

split_file_indices = {
    "train": shuffled_indices[:train_end],

    "validation": shuffled_indices[
        train_end:validation_end
    ],

    "test": shuffled_indices[
        validation_end:
    ],
}


# ============================================================
# 분할별 학습 구간 생성
# ============================================================

def create_split(split_name, file_indices):
    imu_windows = []
    sensor_orientation_windows = []

    target_rotation_windows = []
    target_position_windows = []

    source_names = []
    start_frames = []

    rejected_windows = 0

    sensor_names = None
    target_joint_names = None

    for file_index in file_indices:
        dataset_path = dataset_files[file_index]
        data = np.load(dataset_path)

        imu_data = data["imu_data"]

        sensor_orientations = data[
            "sensor_orientations"
        ]

        target_rotations = data[
            "target_joint_rotations"
        ]

        target_positions = data[
            "target_joint_positions"
        ]

        source_file = str(
            data["source_file"]
        )

        sensor_names = data["sensor_names"]
        target_joint_names = data[
            "target_joint_names"
        ]

        # 모든 데이터의 프레임 수가 같은지 확인
        frame_counts = [
            len(imu_data),
            len(sensor_orientations),
            len(target_rotations),
            len(target_positions),
        ]

        if len(set(frame_counts)) != 1:
            print(
                "프레임 수 불일치로 파일 제외:",
                dataset_path.name,
                frame_counts,
            )
            continue

        # 처음과 끝의 경계 영향을 제외한 범위
        start_limit = BOUNDARY_TRIM
        end_limit = (
            len(imu_data) - BOUNDARY_TRIM
        )

        for start in range(
            start_limit,
            end_limit - WINDOW_SIZE + 1,
            WINDOW_STRIDE,
        ):
            end = start + WINDOW_SIZE

            imu_window = imu_data[start:end]

            orientation_window = (
                sensor_orientations[start:end]
            )

            rotation_window = (
                target_rotations[start:end]
            )

            position_window = (
                target_positions[start:end]
            )

            # IMU의 앞 3개 값은 가속도,
            # 뒤 3개 값은 각속도
            acceleration = imu_window[:, :, :3]
            gyroscope = imu_window[:, :, 3:]

            acceleration_magnitude = np.linalg.norm(
                acceleration,
                axis=2,
            )

            gyroscope_magnitude = np.linalg.norm(
                gyroscope,
                axis=2,
            )

            # NaN 또는 무한대 확인
            invalid_number = (
                np.isnan(imu_window).any()
                or np.isinf(imu_window).any()

                or np.isnan(
                    orientation_window
                ).any()

                or np.isinf(
                    orientation_window
                ).any()

                or np.isnan(
                    rotation_window
                ).any()

                or np.isinf(
                    rotation_window
                ).any()

                or np.isnan(
                    position_window
                ).any()

                or np.isinf(
                    position_window
                ).any()
            )

            # 지나치게 큰 가속도 확인
            excessive_acceleration = (
                acceleration_magnitude.max()
                > MAX_ACCELERATION
            )

            # 지나치게 큰 각속도 확인
            excessive_gyroscope = (
                gyroscope_magnitude.max()
                > MAX_GYROSCOPE
            )

            if (
                invalid_number
                or excessive_acceleration
                or excessive_gyroscope
            ):
                rejected_windows += 1
                continue

            imu_windows.append(
                imu_window.astype(np.float32)
            )

            sensor_orientation_windows.append(
                orientation_window.astype(
                    np.float32
                )
            )

            target_rotation_windows.append(
                rotation_window.astype(
                    np.float32
                )
            )

            target_position_windows.append(
                position_window.astype(
                    np.float32
                )
            )

            source_names.append(source_file)
            start_frames.append(start)

    if not imu_windows:
        raise ValueError(
            f"{split_name}에 사용할 구간이 없습니다."
        )

    # 리스트를 하나의 학습 배열로 변환
    imu_windows = np.stack(
        imu_windows
    )

    sensor_orientation_windows = np.stack(
        sensor_orientation_windows
    )

    target_rotation_windows = np.stack(
        target_rotation_windows
    )

    target_position_windows = np.stack(
        target_position_windows
    )

    source_names = np.array(
        source_names
    )

    start_frames = np.array(
        start_frames,
        dtype=np.int32,
    )

    # ========================================================
    # 결과 저장
    # ========================================================

    output_path = (
        OUTPUT_DIRECTORY / f"{split_name}.npz"
    )

    np.savez_compressed(
        output_path,

        # 모델 입력 후보
        imu=imu_windows,
        sensor_orientations=(
            sensor_orientation_windows
        ),

        # 자세 정답
        target_rotations=(
            target_rotation_windows
        ),
        target_positions=(
            target_position_windows
        ),

        # 데이터 출처
        source_files=source_names,
        start_frames=start_frames,

        # 메타데이터
        sensor_names=sensor_names,
        target_joint_names=target_joint_names,
        frame_rate=np.array(FRAME_RATE),
        window_size=np.array(WINDOW_SIZE),
        window_stride=np.array(WINDOW_STRIDE),
    )

    print()
    print("분할:", split_name)
    print("원본 파일 수:", len(file_indices))
    print("생성 구간:", len(imu_windows))
    print("제외 구간:", rejected_windows)

    print(
        "IMU 크기:",
        imu_windows.shape,
    )

    print(
        "센서 방향 크기:",
        sensor_orientation_windows.shape,
    )

    print(
        "정답 회전 크기:",
        target_rotation_windows.shape,
    )

    print(
        "정답 위치 크기:",
        target_position_windows.shape,
    )

    print("저장:", output_path)


# ============================================================
# Train / Validation / Test 생성
# ============================================================

for split_name, file_indices in (
    split_file_indices.items()
):
    create_split(
        split_name,
        file_indices,
    )


print()
print("학습 구간 생성 완료")