import inspect

import numpy as np


# Python 3.11 호환 처리
if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec

# 최신 NumPy에서 삭제된 옛 자료형 이름 복원
np.bool = np.bool_
np.int = int
np.float = float
np.complex = complex
np.object = object
np.unicode = str
np.str = str

import smplx


model = smplx.create(
    model_path="models",
    model_type="smpl",
    gender="neutral",
    ext="pkl",
)

output = model(return_verts=True)

print("SMPL 모델 로딩 성공")
print("관절 좌표 크기:", output.joints.shape)
print("인체 정점 크기:", output.vertices.shape)