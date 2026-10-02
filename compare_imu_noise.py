import matplotlib.pyplot as plt
import numpy as np


clean = np.load("outputs/first_motion_virtual_imu_50hz.npz")
noisy = np.load("outputs/first_motion_virtual_imu_noisy_50hz.npz")

time = clean["timestamps"]

# 0번 센서인 가슴 센서 사용
clean_acc = clean["accelerometer"][:, 0]
noisy_acc = noisy["accelerometer"][:, 0]

clean_gyro = clean["gyroscope"][:, 0]
noisy_gyro = noisy["gyroscope"][:, 0]

# XYZ 벡터의 크기 계산
clean_acc_norm = np.linalg.norm(clean_acc, axis=1)
noisy_acc_norm = np.linalg.norm(noisy_acc, axis=1)

clean_gyro_norm = np.linalg.norm(clean_gyro, axis=1)
noisy_gyro_norm = np.linalg.norm(noisy_gyro, axis=1)

fig, axes = plt.subplots(2, 1, figsize=(10, 7))

axes[0].plot(time, clean_acc_norm, label="Clean")
axes[0].plot(time, noisy_acc_norm, label="Noisy", alpha=0.8)
axes[0].set_title("Chest Accelerometer")
axes[0].set_ylabel("Acceleration (m/s²)")
axes[0].grid(True)
axes[0].legend()

axes[1].plot(time, clean_gyro_norm, label="Clean")
axes[1].plot(time, noisy_gyro_norm, label="Noisy", alpha=0.8)
axes[1].set_title("Chest Gyroscope")
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Angular velocity (rad/s)")
axes[1].grid(True)
axes[1].legend()

plt.tight_layout()
plt.savefig("outputs/imu_noise_comparison.png", dpi=150)
plt.show()

print("그래프 저장 완료: outputs/imu_noise_comparison.png")