% ppg_imu_from_txt_and_preprocess.m

function [ppg_hp, ppg_bpHR, imuOut, fs_out, mains_out] = ppg_imu_from_txt_and_preprocess(txtFile, fs, mains)
%PPG_IMU_FROM_TXT_AND_PREPROCESS  读取你的 txt（首行中文表头），预处理 PPG 与 IMU，
% 并导出 preprocessed_for_python.mat 供 Python 做"心搏模板相减 + ANC"。
%
% 用法：
%   % 方式1：带参数调用
%   ppg_imu_from_txt_and_preprocess('your_measurement.txt', 100, 50);
%
%   % 方式2：不带参数（需手改下面的默认路径或弹框选择）
%   ppg_imu_from_txt_and_preprocess();
%
% 输出：
%   ppg_hp    : 陷波+高通后的 PPG
%   ppg_bpHR  : 0.8–5 Hz 带通（心搏检测用）
%   imuOut    : 结构体（acc_bp, 可选 gyro_bp, acc_mag, 可选 gyro_mag, jerk_acc, stEnergy）
%   fs_out    : 采样率
%   mains_out : 工频
%
% 说明：
%   - txt 首行为中文表头：绿光---红外光---红光---X轴---Y轴---Z轴
%   - 数据从第二行开始：第1列 PPG_绿光，第2列 PPG_红外，第3列 PPG_红光，第4-6列 三轴加速度

    % ======================== 参数与输入 ========================
    if nargin < 1 || isempty(txtFile)
        % 如需弹窗选择，可取消下一行注释并注释默认路径
        % [fName, fPath] = uigetfile('*.txt', '选择带中文表头的 PPG/IMU txt');
        % if isequal(fName,0), error('未选择文件'); end
        % txtFile = fullfile(fPath, fName);
        txtFile = './hyx_data/喉咙-吞咽6次间隔10秒.txt'; % <- 修改为你的文件路径
    end
    if nargin < 2 || isempty(fs),    fs = 100; end
    if nargin < 3 || isempty(mains), mains = 50; end

    fs_out    = fs;
    mains_out = mains;

    % ======================== 读取数据 ==========================
    % 跳过首行中文表头
    M = readmatrix(txtFile, 'NumHeaderLines', 1);

    % 列映射：按你提供的格式
    ppg_green = M(:,1);
    ppg_ir    = M(:,2);
    ppg_red   = M(:,3);
    acc       = M(:,4:6);   % X, Y, Z
    gyro      = [];         % 本文件无陀螺

    % 选择要处理的 PPG 通道（默认绿光；可改 ppg_ir / ppg_red）
    ppg = ppg_green;

    % ======================== 预处理主流程 ======================
    [ppg_hp, ppg_bpHR, imuOut] = ppg_imu_preprocess_100Hz(ppg, acc, gyro, fs, mains);

    % 导出给 Python 使用
    save('preprocessed_for_python.mat', 'ppg_hp', 'ppg_bpHR', 'imuOut', 'fs', 'mains');
    fprintf('[OK] 已导出 preprocessed_for_python.mat（供 Python 下一步使用）\n');
end


% =================================================================
%                           子函数区域
% =================================================================

function [ppg_hp, ppg_bpHR, imuOut] = ppg_imu_preprocess_100Hz(ppg, acc, gyro, fs, mains)
%PPG_IMU_PREPROCESS_100HZ  工频陷波 + 高通去漂移 + 心搏带通；IMU 高通/带通/特征

    if nargin < 4 || isempty(fs),    fs = 100; end
    if nargin < 5 || isempty(mains), mains = 50; end
    if nargin < 3, gyro = []; end

    ppg = ppg(:);
    if size(acc,2) ~= 3
        error('acc size must be Nx3');
    end
    hasGyro = ~isempty(gyro) && size(gyro,2) == 3;

    % 1) PPG：工频陷波（仅对 f0 < fs/2 的谐波），避免 W0==1（Nyquist）时报错
    ppg_notch = notch_mains_all(ppg, fs, mains, 30);

    % 2) PPG：高通 0.1 Hz 去漂移
    [bh, ah] = butter(2, 0.1/(fs/2), 'high');
    ppg_hp   = filtfilt(bh, ah, ppg_notch);

    % 3) PPG：心搏带通 0.8–5 Hz（峰检/HR）
    [bb, ab] = butter(3, [0.8 5]/(fs/2), 'bandpass');
    ppg_bpHR = filtfilt(bb, ab, ppg_hp);

    % 4) IMU：高通 ~0.3 Hz 去重力 + 带通 0.5–15 Hz
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

    % 5) IMU 特征：模长、jerk、短时能量（0.5 s 窗）
    acc_mag = sqrt(sum(acc_bp.^2, 2));
    if hasGyro
        gyro_mag = sqrt(sum(gyro_bp.^2, 2));
    else
        gyro_mag = [];
    end
    jerk_acc = [zeros(1,3); diff(acc_bp)];

    wlen = round(0.5*fs);
    if mod(wlen, 2) == 0, wlen = wlen + 1; end
    stEnergy.acc  = moving_energy(acc_mag,  wlen);
    if hasGyro
        stEnergy.gyro = moving_energy(gyro_mag, wlen);
    else
        stEnergy.gyro = [];
    end

    % 输出
    imuOut.acc_bp   = acc_bp;
    imuOut.acc_mag  = acc_mag;
    imuOut.jerk_acc = jerk_acc;
    imuOut.stEnergy = stEnergy;
    if hasGyro
        imuOut.gyro_bp  = gyro_bp;
        imuOut.gyro_mag = gyro_mag;
    end
end


function y = notch_mains_all(x, fs, mains, Q)
%NOTCH_MAINS_ALL 对 mains 及其谐波（仅 f0 < fs/2）做 IIR 陷波，避免 W0==1
%   - W0, BW 为相对 Nyquist 的归一化频率
%   - Q ~ 30（可调）

    if nargin < 4, Q = 30; end
    y = x(:);

    nyq = fs/2;
    % 仅保留 f0 < nyq 的谐波；避免在 Nyquist 处设计陷波
    maxH = floor((nyq - 1e-9) / mains);

    for k = 1:maxH
        f0 = k * mains;     % Hz
        W0 = f0 / nyq;      % 归一化中心频率（0~1）
        BW = W0 / Q;        % 归一化带宽

        [b, a] = iirnotch(W0, BW);
        y = filtfilt(b, a, y);
    end
end


function y = moving_energy(x, w)
%MOVING_ENERGY 简单短时能量：平方后滑动平均（线性相位用 filtfilt）
    x2  = x.^2;
    win = ones(w,1) / w;
    y   = filtfilt(win, 1, x2);
end


