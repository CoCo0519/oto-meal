% 确保 batch_export_approx_txt.m 在路径上
% Make sure batch_export_approx_txt.m is on the MATLAB path.

% ===== D: 盘 =====
inDirD  = 'D:\Project-Swallow\hyx_data';
outDirD = 'D:\Project-Swallow\denoised_hyx_data';

batch_export_approx_txt(inDirD, outDirD, ...
    'HeaderLines', 1, ...   % 首行表头
    'Recursive',   true, ...
    'SaveV73',     false);  % 如需 v7.3（HDF5）改为 true

% ===== G: 盘 =====
inDirG  = 'G:\Project-Swallow\hyx_data';
outDirG = 'G:\Project-Swallow\denoised_hyx_data';

batch_export_approx_txt(inDirG, outDirG, ...
    'HeaderLines', 1, ...
    'Recursive',   true, ...
    'SaveV73',     false);

% ===== 快速自检（任选一个导出的 MAT 检查 approx 与 A7 是否一致）=====
S = dir(fullfile(outDirG, '**', '*_denoise.mat'));
if ~isempty(S)
    f = fullfile(S(1).folder, S(1).name);
    disp("Check: " + string(f));
    L = whos('-file', f); disp(L);
    load(f, 'approx','data','data_raw','names','meta');
    % 抽一列做一致性验证
    x  = data_raw(:,1);
    y  = wdenoise(x, meta.params.level, ...
                  Wavelet=meta.params.wavelet, ...
                  DenoisingMethod=meta.params.method, ...
                  ThresholdRule=meta.params.rule, ...
                  NoiseEstimate=meta.params.noise);
    [c,l] = wavedec(y, meta.params.level, meta.params.wavelet);
    a7    = wrcoef('a', c, l, meta.params.wavelet, meta.params.level);
    fprintf('max|a7 - approx(:,1)| = %.3g\n', max(abs(a7 - approx(:,1))));
end
