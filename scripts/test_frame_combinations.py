import os
import sys
sys.path.insert(0, os.path.abspath('src'))
import numpy as np
import pandas as pd

from navris.eskf import ESKF, NominalState, ESKFConfig, STATE_DIM
from navris.inertial.frames import quat_to_euler, rotvec_to_quat, quat_multiply, wrap_angle_pi

# Load synchronized S1 data
df = pd.read_parquet('data/processed/synchronized/S1_sync.parquet')

# Test segment from t=156s to 250s
init_idx = 1560
eval_df = df.iloc[init_idx:init_idx + 1000].copy().reset_index(drop=True)

# Test different gyro mappings:
# In all cases:
# speed = phone_gps_speed_mps * 3.6 (the 3.6x unit fix)
v0_speed = float(eval_df['phone_gps_speed_mps'].iloc[0]) * 3.6
init_heading = 5.5235368  # 316.48 deg (azimuth)
v0 = np.array([v0_speed * np.sin(init_heading), v0_speed * np.cos(init_heading), 0.0])
print(f"Corrected v0: {v0.round(3)}, speed: {v0_speed:.2f} m/s (Ref speed: {eval_df['ref_speed_mps'].iloc[0]:.2f} m/s)")

def test_permutation(gyro_map_func, fwd_axis_func, label):
    print(f"\n" + "=" * 60)
    print(f"TESTING: {label}")
    print("=" * 60)

    p0 = np.array([eval_df['phone_gps_east_m'].iloc[0], eval_df['phone_gps_north_m'].iloc[0], eval_df['phone_gps_up_m'].iloc[0]])

    # Base leveling: gravity along +Z
    # q_level: rotates body Z to nav Up [0, 0, 1]
    # For a flat phone, body Z is [0, 0, 1], so q_level is identity
    stat_sub = df.iloc[:30]
    f_mean = np.array([stat_sub['phone_accel_x_mps2'].mean(), stat_sub['phone_accel_y_mps2'].mean(), stat_sub['phone_accel_z_mps2'].mean()])
    f_norm = np.linalg.norm(f_mean)
    up_b = f_mean / f_norm
    up_n = np.array([0.0, 0.0, 1.0])

    # Leveling quaternion
    v = np.cross(up_b, up_n)
    s = np.dot(up_b, up_n)
    if s < -0.999999:
        q_level = np.array([0.0, 1.0, 0.0, 0.0])
    else:
        q_level = np.array([1.0 + s, v[0], v[1], v[2]])
        q_level /= np.linalg.norm(q_level)

    # Attitude heading alignment:
    # Azimuth psi: forward vector in ENU is [sin(psi), cos(psi), 0]
    # If body forward is fwd_b:
    # We want C_b_n @ fwd_b = [sin(psi), cos(psi), 0]
    # Let's test standard init vs +Y forward init:
    q0 = fwd_axis_func(q_level, init_heading)

    init_state = NominalState(
        t=float(eval_df['time_s'].iloc[0]),
        p=p0, v=v0.copy(), q=q0,
        ba=np.zeros(3), bg=np.zeros(3)
    )
    init_cov = np.diag([
        25.0, 25.0, 100.0,
        4.0, 4.0, 1.0,
        0.01, 0.01, 0.05,
        0.04, 0.04, 0.04,
        1e-4, 1e-4, 1e-4
    ])
    config = ESKFConfig(
        sigma_acc=0.20, sigma_gyr=0.02,
        sigma_acc_bias=1e-3, sigma_gyr_bias=1e-4,
        gnss_pos_std_horiz=3.0, gnss_pos_std_vert=10.0,
        chi2_threshold_pos=16.27
    )
    eskf = ESKF(init_state, init_cov, config, lat_deg=52.4, alt_m=100.0)

    novel_seen = 0
    accepted = 0

    for i in range(len(eval_df)):
        t = float(eval_df['time_s'].iloc[i])
        if i > 0:
            dt = t - float(eval_df['time_s'].iloc[i - 1])
            fb = np.array([eval_df['phone_accel_x_mps2'].iloc[i-1], eval_df['phone_accel_y_mps2'].iloc[i-1], eval_df['phone_accel_z_mps2'].iloc[i-1]])
            raw_wb = np.array([eval_df['phone_gyro_x_radps'].iloc[i-1], eval_df['phone_gyro_y_radps'].iloc[i-1], eval_df['phone_gyro_z_radps'].iloc[i-1]])
            wb = gyro_map_func(raw_wb)
            eskf.predict(fb, wb, dt)

        if bool(eval_df['phone_gps_is_new_fix'].iloc[i]):
            novel_seen += 1
            z_pos = np.array([eval_df['phone_gps_east_m'].iloc[i], eval_df['phone_gps_north_m'].iloc[i], eval_df['phone_gps_up_m'].iloc[i]])
            acc_m = max(1.0, float(eval_df['phone_gps_accuracy_m'].iloc[i]))
            pos_cov = np.diag([acc_m**2, acc_m**2, (acc_m * 3.0)**2])
            acc = eskf.update_gnss_pos(z_pos, pos_cov=pos_cov, is_new_fix=True, timestamp=t)
            inv = eskf.innovations[-1]
            if acc:
                accepted += 1
            r, p, y = np.degrees(quat_to_euler(eskf.state.q))
            ref_p = np.array([eval_df['ref_east_m'].iloc[i], eval_df['ref_north_m'].iloc[i], eval_df['ref_up_m'].iloc[i]])
            h_err = np.linalg.norm(eskf.state.p[:2] - ref_p[:2])
            if novel_seen <= 5:
                print(f"Fix {novel_seen-1} (t={t:.1f}s): Acc={acc}, NIS={inv.mahalanobis_sq:.2f}, ResNorm={np.linalg.norm(inv.residual):.2f}m, HErr={h_err:.2f}m, Pitch={p:.2f}°, Roll={r:.2f}°")

    print(f"Result: {accepted}/{novel_seen} fixes accepted.")

# Define forward axis functions:
def fwd_is_X(q_level, heading):
    # theta = pi/2 - heading
    theta = np.pi / 2.0 - heading
    q_yaw = np.array([np.cos(0.5 * theta), 0.0, 0.0, np.sin(0.5 * theta)])
    return quat_multiply(q_yaw, q_level)

def fwd_is_Y(q_level, heading):
    # If body +Y is forward, in polar angle +Y starts at pi/2.
    # To rotate +Y to polar angle theta = pi/2 - heading,
    # rotation angle is theta - pi/2 = -heading!
    theta = -heading
    q_yaw = np.array([np.cos(0.5 * theta), 0.0, 0.0, np.sin(0.5 * theta)])
    return quat_multiply(q_yaw, q_level)

def fwd_is_negY(q_level, heading):
    # -Y is forward
    theta = np.pi - heading
    q_yaw = np.array([np.cos(0.5 * theta), 0.0, 0.0, np.sin(0.5 * theta)])
    return quat_multiply(q_yaw, q_level)

def fwd_is_diag(q_level, heading):
    # [1, -1] / sqrt(2) is forward (angle -pi/4)
    # theta - (-pi/4) = pi/2 - heading -> rotation = 3pi/4 - heading
    rot = (3.0 * np.pi / 4.0) - heading
    q_yaw = np.array([np.cos(0.5 * rot), 0.0, 0.0, np.sin(0.5 * rot)])
    return quat_multiply(q_yaw, q_level)

# Permutation 1: raw [gx, gy, gz] unchanged
test_permutation(lambda w: w, fwd_is_X, "1. Raw Gyro [gx, gy, gz], +X forward (Baseline setup with speed*3.6)")

# Permutation 2: gyro vertical is gy: [gx, gz, gy], +X forward
test_permutation(lambda w: np.array([w[0], w[2], w[1]]), fwd_is_X, "2. Gyro [gx, gz, gy] (Z_accel = gy), +X forward")

# Permutation 3: gyro vertical is gy: [gx, -gz, gy], +X forward
test_permutation(lambda w: np.array([w[0], -w[2], w[1]]), fwd_is_X, "3. Gyro [gx, -gz, gy], +X forward")

# Permutation 4: gyro vertical is gy: [gz, gx, gy], +X forward
test_permutation(lambda w: np.array([w[2], w[0], w[1]]), fwd_is_X, "4. Gyro [gz, gx, gy], +X forward")

# Permutation 5: gyro vertical is gy: [gx, -gz, gy], +Y forward
test_permutation(lambda w: np.array([w[0], -w[2], w[1]]), fwd_is_Y, "5. Gyro [gx, -gz, gy], +Y forward")

# Permutation 6: gyro vertical is gy: [gx, -gz, gy], diag forward
test_permutation(lambda w: np.array([w[0], -w[2], w[1]]), fwd_is_diag, "6. Gyro [gx, -gz, gy], [1, -1] diag forward")
