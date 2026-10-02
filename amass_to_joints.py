import inspect
from pathlib import Path

import numpy as np
import torch


# Python 3.11 및 최신 NumPy 호환 처리
if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec

np.bool = np.bool_
np.int = int
np.float = float
np.complex = complex
np.object = object
np.unicode = str
np.str = str

import smplx


# 사용할 AMASS 동작 선택
motion_path = Path(
    "synthetic_rehab_motions/"
    "rehab_048_scaption_right_140deg_8sec.npz"
)

motion = np.load(motion_path)

poses_amass = motion["poses"].astype(np.float32)
translations = motion["trans"].astype(np.float32)
frame_rate = float(motion["mocap_framerate"])
gender = str(motion["gender"]).lower()

if gender not in ("male", "female"):
    gender = "neutral"

frame_count = len(poses_amass)

print("입력 파일:", motion_path)
print("성별 모델:", gender)
print("프레임 수:", frame_count)
print("프레임레이트:", frame_rate)


# AMASS의 SMPL+H 자세를 일반 SMPL 자세로 변환
# 앞 66개 값은 루트와 몸 관절 22개의 회전값
smpl_pose = np.zeros(
    (frame_count, 72),
    dtype=np.float32,
)

smpl_pose[:, :66] = poses_amass[:, :66]


# 성별에 맞는 SMPL 모델 불러오기
model = smplx.create(
    model_path="models",
    model_type="smpl",
    gender=gender,
    ext="pkl",
    num_betas=10,
)


# NumPy → PyTorch
pose_tensor = torch.tensor(
    smpl_pose,
    dtype=torch.float32,
)

translation_tensor = torch.tensor(
    translations,
    dtype=torch.float32,
)

betas_tensor = torch.tensor(
    motion["betas"][:10],
    dtype=torch.float32,
).unsqueeze(0).repeat(frame_count, 1)


# 모든 프레임의 3D 관절 좌표 계산
with torch.no_grad():
    output = model(
        global_orient=pose_tensor[:, :3],
        body_pose=pose_tensor[:, 3:],
        betas=betas_tensor,
        transl=translation_tensor,
        return_verts=False,
    )

# SMPL 기본 관절 24개만 사용
joints = output.joints[:, :24].cpu().numpy()


# 결과 저장
output_directory = Path("outputs")
output_directory.mkdir(exist_ok=True)

output_path = output_directory / "first_motion_joints.npz"

np.savez(
    output_path,
    joints=joints,
    frame_rate=frame_rate,
    source_file=str(motion_path),
    gender=gender,
)

print("3D 관절 좌표 크기:", joints.shape)
print("저장 완료:", output_path)