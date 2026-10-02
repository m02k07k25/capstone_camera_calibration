from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


FRAME_RATE = 120
DURATION = 8.0
FRAME_COUNT = int(FRAME_RATE * DURATION)

OUTPUT_DIRECTORY = Path(
    "synthetic_dataset_v1"
)

OUTPUT_DIRECTORY.mkdir(exist_ok=True)


def create_shoulder_flexion(
    filename,
    target_angle_degrees,
    movement_duration,
    label,
    fault_type,
    trunk_compensation_degrees=0.0,
):
    timestamps = (
        np.arange(FRAME_COUNT) / FRAME_RATE
    )

    shoulder_angles = np.zeros(
        FRAME_COUNT,
        dtype=np.float64,
    )

    start_time = 1.0
    raise_end_time = (
        start_time + movement_duration
    )

    hold_end_time = raise_end_time + 1.0

    lower_end_time = (
        hold_end_time + movement_duration
    )


    # 어깨 굴곡 각도 생성
    for frame_index, time in enumerate(timestamps):
        # 팔 올리기
        if start_time <= time < raise_end_time:
            progress = (
                time - start_time
            ) / movement_duration

            shoulder_angles[frame_index] = (
                target_angle_degrees
                * 0.5
                * (
                    1.0
                    - np.cos(np.pi * progress)
                )
            )

        # 목표 각도 유지
        elif raise_end_time <= time < hold_end_time:
            shoulder_angles[frame_index] = (
                target_angle_degrees
            )

        # 팔 내리기
        elif hold_end_time <= time < lower_end_time:
            progress = (
                time - hold_end_time
            ) / movement_duration

            shoulder_angles[frame_index] = (
                target_angle_degrees
                * 0.5
                * (
                    1.0
                    + np.cos(np.pi * progress)
                )
            )


    # AMASS와 같은 SMPL+H 자세 배열
    poses = np.zeros(
        (FRAME_COUNT, 156),
        dtype=np.float32,
    )


    # SMPL Y-up 자세를 AMASS Z-up 좌표계로 변환
    root_rotation = Rotation.from_euler(
        "x",
        90,
        degrees=True,
    ).as_rotvec()

    poses[:, 0:3] = root_rotation


    # 왼팔은 몸 옆에 고정
    left_arm_down = Rotation.from_euler(
        "z",
        -90,
        degrees=True,
    )

    poses[:, 16 * 3:16 * 3 + 3] = (
        left_arm_down.as_rotvec()
    )


    # 오른팔 기본 자세
    right_arm_down = Rotation.from_euler(
        "z",
        90,
        degrees=True,
    )


    for frame_index, angle in enumerate(
        shoulder_angles
    ):
        # 오른쪽 어깨 굴곡
        flexion_rotation = Rotation.from_euler(
            "x",
            -angle,
            degrees=True,
        )

        right_shoulder_rotation = (
            flexion_rotation * right_arm_down
        )

        poses[
            frame_index,
            17 * 3:17 * 3 + 3,
        ] = right_shoulder_rotation.as_rotvec()


        # 몸통 보상 동작
        if trunk_compensation_degrees > 0:
            angle_ratio = (
                angle / target_angle_degrees
                if target_angle_degrees > 0
                else 0.0
            )

            trunk_angle = (
                trunk_compensation_degrees
                * angle_ratio
            )

            trunk_rotation = Rotation.from_euler(
                "x",
                trunk_angle,
                degrees=True,
            )

            # SMPL 관절 3: 아래쪽 척추
            poses[
                frame_index,
                3 * 3:3 * 3 + 3,
            ] = trunk_rotation.as_rotvec()


    translations = np.zeros(
        (FRAME_COUNT, 3),
        dtype=np.float32,
    )

    betas = np.zeros(
        16,
        dtype=np.float32,
    )

    output_path = (
        OUTPUT_DIRECTORY / filename
    )

    np.savez(
        output_path,
        poses=poses,
        trans=translations,
        gender=np.array("neutral"),
        mocap_framerate=np.array(FRAME_RATE),
        betas=betas,
        label=np.array(label),
        fault_type=np.array(fault_type),
        target_angle_degrees=np.array(
            target_angle_degrees
        ),
        movement_duration=np.array(
            movement_duration
        ),
        trunk_compensation_degrees=np.array(
            trunk_compensation_degrees
        ),
    )

    print(
        filename,
        "| label:",
        label,
        "| angle:",
        target_angle_degrees,
        "| speed:",
        movement_duration,
        "| trunk:",
        trunk_compensation_degrees,
    )


# 정상 동작 3개
create_shoulder_flexion(
    filename="normal_01.npz",
    target_angle_degrees=140.0,
    movement_duration=2.5,
    label=0,
    fault_type="normal",
)

create_shoulder_flexion(
    filename="normal_02.npz",
    target_angle_degrees=150.0,
    movement_duration=2.0,
    label=0,
    fault_type="normal",
)

create_shoulder_flexion(
    filename="normal_03.npz",
    target_angle_degrees=160.0,
    movement_duration=1.5,
    label=0,
    fault_type="normal",
)


# 가동범위 부족 3개
create_shoulder_flexion(
    filename="limited_rom_01.npz",
    target_angle_degrees=70.0,
    movement_duration=2.0,
    label=1,
    fault_type="limited_rom",
)

create_shoulder_flexion(
    filename="limited_rom_02.npz",
    target_angle_degrees=90.0,
    movement_duration=2.0,
    label=1,
    fault_type="limited_rom",
)

create_shoulder_flexion(
    filename="limited_rom_03.npz",
    target_angle_degrees=110.0,
    movement_duration=2.0,
    label=1,
    fault_type="limited_rom",
)


# 몸통 보상 동작 3개
create_shoulder_flexion(
    filename="trunk_compensation_01.npz",
    target_angle_degrees=150.0,
    movement_duration=2.0,
    label=2,
    fault_type="trunk_compensation",
    trunk_compensation_degrees=10.0,
)

create_shoulder_flexion(
    filename="trunk_compensation_02.npz",
    target_angle_degrees=150.0,
    movement_duration=2.0,
    label=2,
    fault_type="trunk_compensation",
    trunk_compensation_degrees=15.0,
)

create_shoulder_flexion(
    filename="trunk_compensation_03.npz",
    target_angle_degrees=150.0,
    movement_duration=2.0,
    label=2,
    fault_type="trunk_compensation",
    trunk_compensation_degrees=20.0,
)


print()
print("합성 동작 데이터셋 v1 생성 완료")
print("저장 폴더:", OUTPUT_DIRECTORY)
print("전체 파일 수: 9")