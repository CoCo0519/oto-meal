% 运行本脚本前，请确保 batch_wavelet_denoise_txt.m 在 MATLAB 路径上。
% Make sure batch_wavelet_denoise_txt.m is on the MATLAB path before running.

% % 采样格式：首行表头=1；每行 6 列数值（如：绿/红外/红/IMU X/Y/Z）；
% HeaderLines=1; each row has 6 numeric columns (e.g., Green/IR/Red/IMU X/Y/Z).

% —— G 盘路径 —— | G drive paths
inDirD  = 'G:\Project-Swallow\hyx_data';
outDirD = 'G:\Project-Swallow\denoised_hyx_data';

batch_wavelet_denoise_txt(inDirD, outDirD, ...
    'SignalColumns', 1:6, ...
    'HeaderLines', 1, ...
    'Recursive', true, ...
    'SaveV73', false);  % 如需 v7.3（超大文件/兼容 HDF5），改为 true
                        % set true for -v7.3 if very large files/HDF5 needed

% —— G 盘路径 —— | G drive paths
inDirG  = 'G:\Project-Swallow\hyx_data';
outDirG = 'G:\Project-Swallow\denoised_hyx_data';

batch_wavelet_denoise_txt(inDirG, outDirG, ...
    'SignalColumns', 1:6, ...
    'HeaderLines', 1, ...
    'Recursive', true, ...
    'SaveV73', false);

% —— 自检（任选一个导出的 MAT）—— | Quick self-check (pick one exported MAT)
S = dir(fullfile(outDirG, '**', '*_denoise.mat'));
if ~isempty(S)
    testFile = fullfile(S(1).folder, S(1).name);
    disp("Check file: " + string(testFile));
    L = whos('-file', testFile); disp(L);        % 列出变量 / list variables
    load(testFile, 'data','data_raw','names','meta');
    fprintf('size(data) = [%d %d], size(data_raw) = [%d %d]\n', size(data), size(data_raw));
    disp("names:"); disp(names(:)');
    % 与单次 wdenoise 抽查一致性 / spot-check parity with a single-column wdenoise
    y_chk = wdenoise(data_raw(:,1), 7, Wavelet='sym8', DenoisingMethod='Bayes', ThresholdRule='Soft', NoiseEstimate='LevelIndependent');
    fprintf('max|difference| on col1 = %.3g\n', max(abs(y_chk - data(:,1))));
end
