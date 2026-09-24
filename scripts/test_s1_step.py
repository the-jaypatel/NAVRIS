import pandas as pd
import numpy as np
from navris.eskf import ESKF, NominalState, ESKFConfig
from navris.inertial.init import align_attitude_causal

df = pd.read_parquet("data/processed/synchronized/S1_sync.parquet")
causal_history = df.iloc[:3500]
align_res = align_attitude_causal(causal_history, min_stat_samples=30)
init_idx = align_res.init_sample_idx
eval_df = df.iloc[init_idx:].copy().reset_index(drop=True)

p0 = np.array([eval_df["phone_gps_east_m"].iloc[0], eval_df["phone_gps_north_m"].iloc[0], eval_df["phone_gps_up_m"].iloc[0]])
v0 = align_res.initial_velocity_enu
q0 = align_res.q_b_n
b_g0 = align_res.gyro_bias_radps

init_state = NominalState(t=float(eval_df["time_s"].iloc[0]), p=p0, v=v0, q=q0, ba=np.zeros(3), bg=b_g0)
init_cov = np.diag([25.0, 25.0, 100.0, 4.0, 4.0, 1.0, 0.01, 0.01, 0.05, 0.04, 0.04, 0.04, 1e-4, 1e-4, 1e-4])
config = ESKFConfig(
    sigma_acc=0.2,
    sigma_gyr=0.02,
    sigma_acc_bias=1e-3,
    sigma_gyr_bias=1e-4,
    gnss_pos_std_horiz=3.0,
    gnss_pos_std_vert=10.0,
    chi2_threshold_pos=16.27
)
eskf = ESKF(init_state, init_cov, config, lat_deg=52.4, alt_m=100.0)

for i in range(300):
    dt = eval_df["time_s"].iloc[i+1] - eval_df["time_s"].iloc[i]
    fb = np.array([eval_df["phone_accel_x_mps2"].iloc[i], eval_df["phone_accel_y_mps2"].iloc[i], eval_df["phone_accel_z_mps2"].iloc[i]])
    wb = np.array([eval_df["phone_gyro_x_radps"].iloc[i], eval_df["phone_gyro_y_radps"].iloc[i], eval_df["phone_gyro_z_radps"].iloc[i]])
    eskf.predict(fb, wb, dt)

    is_new = bool(eval_df["phone_gps_is_new_fix"].iloc[i])
    z_pos = np.array([eval_df["phone_gps_east_m"].iloc[i], eval_df["phone_gps_north_m"].iloc[i], eval_df["phone_gps_up_m"].iloc[i]])
    acc_m = float(eval_df["phone_gps_accuracy_m"].iloc[i]) if "phone_gps_accuracy_m" in eval_df.columns else 3.0
    pos_cov = np.diag([acc_m**2, acc_m**2, (acc_m * 3.0)**2])
    eskf.update_gnss_pos(z_pos, pos_cov=pos_cov, is_new_fix=is_new, timestamp=eval_df["time_s"].iloc[i])

new_invs = [inv for inv in eskf.innovations if inv.is_new_fix]
print(f"Total new fixes in 30s: {len(new_invs)}")
for idx, inv in enumerate(new_invs):
    print(f"Fix {idx} t={inv.timestamp:.1f}s: accept={inv.accepted} d2={inv.mahalanobis_sq:6.2f} res={np.round(inv.residual, 1)} S_std={np.round(np.sqrt(np.diag(inv.S_cov)), 1)}")
