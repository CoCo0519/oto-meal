% 读取PPG+IMU数据（空格分隔），并命名列
% Read PPG+IMU (space-delimited) and name columns
opts = detectImportOptions('G:\Project-Swallow\hyx_data\喉咙-吞咽每5秒一次共31秒.txt','NumHeaderLines',1);
opts.Delimiter = {' ','\t'};
opts.VariableNames = {'PPG_G','PPG_IR','PPG_R','ACC_X','ACC_Y','ACC_Z'};
T = readtable('G:\Project-Swallow\hyx_data\喉咙-吞咽每5秒一次共31秒.txt', opts);

% 取绿光PPG作为示例，并去均值，减少直流分量
% Use green PPG as example; remove mean to reduce DC
ppg_green = detrend(double(T.PPG_G), 0);  % 0=remove mean

% （可选）设置采样率，若未知先假设100Hz，后续在GUI里修正
% (Optional) Sampling rate; if unknown, start with 100 Hz and adjust in GUI
Fs = 50;

% 打开Wavelet Signal Denoiser应用
% Open the Wavelet Signal Denoiser app
waveletSignalDenoiser
