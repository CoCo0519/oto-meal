%% PPG & (3-axis) IMU preprocessing @100 Hz (accelerometer-only compatible)
% 功能：
% - PPG：工频陷波 -> 高通去漂移 -> （用于心率估计的）心搏带通
% - IMU：支持仅三轴 IMU（仅加速度）。去重力/低频 -> 0.5–15 Hz 带通 -> jerk/能量
% 使用：
% % 若只有三轴加速度：把 gyro 传 []
% [ppg_hp, ppg_bpHR, imuOut] = ppg_imu_preprocess_100Hz(ppg, acc, [], 100, 50);
% % 若有陀螺仪：
% [ppg_hp, ppg_bpHR, imuOut] = ppg_imu_preprocess_100Hz(ppg, acc, gyro, 100, 50);
% 输入：
% ppg : Nx1 原始 PPG 序列
% acc : Nx3 加速度 (x,y,z)
% gyro : Nx3 陀螺仪 (x,y,z)；若没有，传 []
% fs : 采样率（建议 100）
% mains: 工频（50 或 60）
% 输出：
% ppg_hp : 高通+陷波后的 PPG（去漂移、去工频）
% ppg_bpHR : 0.8–5 Hz 带通（用于心搏检测/HR 估计）
% imuOut : 结构体，包含 acc_bp, （可选）gyro_bp, acc_mag, （可选）gyro_mag, jerk_acc, stEnergy 等
%
% 说明：心搏模板相减与 ANC 放在 Python 代码实现。

function [ppg_hp, ppg_bpHR, imuOut] = ppg_imu_preprocess_100Hz(ppg, acc, gyro, fs, mains)
if nargin < 4 || isempty(fs), fs = 100; end
if nargin < 5 || isempty(mains), mains = 50; end
if nargin < 3, gyro = []; end

ppg = ppg(:);
if size(acc,2)~=3, error('acc size must be Nx3'); end
hasGyro = ~isempty(gyro) && size(gyro,2)==3;

%% 1) PPG: 工频陷波 + 高通去漂移 (0.1 Hz)
ppg_notch = notch_filter(ppg, fs, mains);
[bh, ah] = butter(2, 0.1/(fs/2), 'high');
ppg_hp = filtfilt(bh, ah, ppg_notch);

%% 2) PPG: 心搏带通 (0.8–5 Hz) 仅用于 HR 提取/峰检
[bb, ab] = butter(3, [0.8 5]/(fs/2), 'bandpass');
ppg_bpHR = filtfilt(bb, ab, ppg_hp);

%% 3) IMU: 去重力/低频 (高通 ~0.3 Hz) + 带通 0.5–15 Hz
[bhp, ahp] = butter(2, 0.3/(fs/2), 'high');
[bb1, ab1] = butter(3, [0.5 15]/(fs/2), 'bandpass');

acc_hp = filtfilt(bhp, ahp, acc);
acc_bp = filtfilt(bb1, ab1, acc_hp);

if hasGyro
gyro_hp = filtfilt(bhp, ahp, gyro);
gyro_bp = filtfilt(bb1, ab1, gyro_hp);
else
gyro_bp = [];
end

%% 4) IMU 特征：模长、jerk（仅加速度必有）、短时能量
acc_mag = sqrt(sum(acc_bp.^2, 2));
if hasGyro
gyro_mag = sqrt(sum(gyro_bp.^2, 2));
else
gyro_mag = [];
end

% jerk_acc = 加速度一阶差分（补齐长度）
jerk_acc = [zeros(1,3); diff(acc_bp)];

% 短时能量（窗口约 0.5 s）
wlen = round(0.5*fs); if mod(wlen,2)==0, wlen=wlen+1; end
stEnergy.acc = moving_energy(acc_mag, wlen);
if hasGyro
stEnergy.gyro = moving_energy(gyro_mag, wlen);
else
stEnergy.gyro = [];
end

%% 输出结构
imuOut.acc_bp = acc_bp;
imuOut.acc_mag = acc_mag;
imuOut.jerk_acc = jerk_acc;
imuOut.stEnergy = stEnergy;
if hasGyro
imuOut.gyro_bp = gyro_bp;
imuOut.gyro_mag = gyro_mag;
end
end

%% ----------------- 辅助函数 -----------------
function y = moving_energy(x, w)
% 简单短时能量（平方后滑窗平均）
x2 = x.^2;
win = ones(w,1)/w;
y = filtfilt(win, 1, x2);
end

function y = notch_filter(x, fs, mains)
% IIR 陷波，Q 约 30，抑制 mains 及其谐波（到 Nyquist）
if nargin<3, mains=50; end
end
