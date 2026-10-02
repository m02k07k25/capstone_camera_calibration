from pathlib import Path

import numpy as np


npz_files = list(Path("ACCAD").rglob("*.npz"))

if not npz_files:
    raise FileNotFoundError("ACCAD 폴더에서 npz 파일을 찾지 못했습니다.")

file_path = npz_files[0]
data = np.load(file_path)

print("선택한 파일:", file_path)
print("데이터 항목:", data.files)

for key in data.files:
    value = data[key]
    print(f"{key}: shape={value.shape}, dtype={value.dtype}")