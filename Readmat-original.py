# run_pipeline_from_mat.py

import numpy as np
import scipy.io as sio
from anc_template_pipeline_100Hz import run_pipeline

def _arr_or_none(val):
    if val is None:
        return None
    arr = np.array(val, dtype=float)
    if arr.size == 0:
        return None
    return arr

# 关键：squeeze_me=True + struct_as_record=False
mat = sio.loadmat('preprocessed_for_python.mat', squeeze_me=True, struct_as_record=False)

# 基本量
ppg   = np.array(mat['ppg_hp']).ravel()
fs    = int(np.array(mat['fs']).squeeze())
mains = int(np.array(mat['mains']).squeeze())

# MATLAB struct -> mat_struct（属性访问）
imuOut = mat['imuOut']

# 必选：acc_bp（没有就报错）
acc_bp = _arr_or_none(getattr(imuOut, 'acc_bp', None))
if acc_bp is None:
    raise ValueError("imuOut.acc_bp 缺失或为空，检查 MATLAB 预处理导出的内容。")

# 可选：gyro_bp（可能不存在/为空）
gyro_bp = _arr_or_none(getattr(imuOut, 'gyro_bp', None))

# 直接进入模板相减 + ANC；gyro_bp 为 None 时会自动仅用加速度参考
out = run_pipeline(ppg, acc_bp, gyro=gyro_bp, fs=fs, mains=mains, make_demo_plot=True)

print("done. keys:", list(out.keys()))
