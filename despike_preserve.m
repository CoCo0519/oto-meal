function [X_clean, events, featTbl, cfg] = despike_preserve(X, fs, varargin)
% 去毛刺并保留显著变化；输出去噪矩阵、事件边界、特征表
% Despike while preserving salient changes; outputs denoised matrix, events, features.
%
% X : N×C 信号矩阵（每列一个通道）
% fs: 采样率 (Hz)
%
% 可选参数 / Name-Value:
% 'HampelWin'   (default: round(0.15*fs))   % Hampel 窗长（秒转点）
% 'HampelK'     (default: 3)                % MAD 阈
% 'EdgeQuantile'(default: 0.98)             % 导数绝对值分位做边缘阈
% 'BfWin'       (default: round(0.20*fs))   % 双边滤波窗口（点）
% 'SigmaS'      (default: 0.10)             % 时间权（窗口内归一化到[-1,1]再乘）
% 'SigmaR'      (default: 2.5)              % 幅值权（单位=信号robust标准差）
% 'BfPasses'    (default: 1)                % 双边滤波遍数（非边缘区）
% 'DoWavelet'   (default: true)             % 是否做温和小波清理
% 'WavName'     (default: 'sym8')
% 'WavLevel'    (default: 2)
% 'MaxChanges'  (default: 8)                % 每通道最大变化点数 for findchangepts
%
% 输出：
% X_clean : 去毛刺且边缘保留的结果（N×C）
% events  : 每列一个结构，含 changePts、maskEdge、indicesSpike
% featTbl : 聚合特征表（每列一组行），含段落起止、峰/谷、坡度等
% cfg     : 实际使用的配置

p = inputParser;
addParameter(p,'HampelWin',[]);
addParameter(p,'HampelK',3);
addParameter(p,'EdgeQuantile',0.98);
addParameter(p,'BfWin',[]);
addParameter(p,'SigmaS',0.10);
addParameter(p,'SigmaR',2.5);
addParameter(p,'BfPasses',1);
addParameter(p,'DoWavelet',true);
addParameter(p,'WavName','sym8');
addParameter(p,'WavLevel',2);
addParameter(p,'MaxChanges',8);
parse(p,varargin{:});
cfg = p.Results;

[N,C] = size(X);
if isempty(cfg.HampelWin), cfg.HampelWin = max(3, round(0.15*fs)); end
if isempty(cfg.BfWin),    cfg.BfWin    = max(5, round(0.20*fs)); end

% 预计算各列的 robust std，用于 SigmaR 的幅值尺度
robStd = zeros(1,C);
for c=1:C
    medc = median(X(:,c),'omitnan');
    robStd(c) = 1.4826 * median(abs(X(:,c)-medc),'omitnan'); % MAD->std
    if robStd(c) < eps, robStd(c) = std(X(:,c)); end
    if robStd(c) < eps, robStd(c) = 1; end
end

X1 = X; idxSpike = cell(1,C);
% 1) Hampel 去孤立尖刺
for c=1:C
    [x_h, ~, outliers] = hampel(X(:,c), cfg.HampelWin, cfg.HampelK);
    X1(:,c) = x_h;
    idxSpike{c} = find(outliers);
end

% 2) 计算导数，构建边缘掩码
X2 = X1;
edgeMask = false(N,C);
for c=1:C
    dx = gradient(X1(:,c));                 % 近似一阶导
    thr = quantile(abs(dx), cfg.EdgeQuantile);
    edgeMask(:,c) = abs(dx) >= thr;         % 强变化区
    % 可视化/验证时可平滑 mask
    edgeMask(:,c) = movmax(edgeMask(:,c), round(0.05*fs));
end

% 3) 非边缘区做 1D 双边滤波
X3 = X2;
for c=1:C
    s = X2(:,c);
    notEdge = ~edgeMask(:,c);
    s2 = bilateral1d_masked(s, notEdge, cfg.BfWin, cfg.SigmaS, cfg.SigmaR*robStd(c), cfg.BfPasses);
    X3(:,c) = s2;
end

% 4) 可选：温和小波软阈（保护边缘：仅对非边缘样本替换）
X4 = X3;
if cfg.DoWavelet
    for c=1:C
        y = wdenoise(X3(:,c), cfg.WavLevel, ...
            'Wavelet', cfg.WavName, ...
            'DenoisingMethod','Bayes', ...
            'ThresholdRule','Soft', ...
            'NoiseEstimate','LevelIndependent');
        ne = ~edgeMask(:,c);         % 仅在非边缘区采纳
        X4(ne,c) = y(ne);
    end
end
X_clean = X4;

% 5) 事件检测与特征
events = cell(1,C);
allFeat = [];
for c=1:C
    x = X_clean(:,c);
    try
        % 变化点（均值+斜率）联合尝试
        cps1 = findchangepts(x,'Statistic','mean','MaxNumChanges',cfg.MaxChanges,'MinThreshold',1);
        cps2 = findchangepts(x,'Statistic','linear','MaxNumChanges',cfg.MaxChanges,'MinThreshold',1);
        cps  = unique([cps1(:); cps2(:)]);
    catch
        cps = [];
    end
    % 由变化点得到区段
    segs = [1; cps(:); N];
    feat = table();
    for sidx = 1:numel(segs)-1
        a = segs(sidx); b = segs(sidx+1);
        seg = x(a:b);
        [mx, imx] = max(seg); [mn, imn] = min(seg);
        slope = (x(b) - x(a)) / max(1,(b-a)/fs);
        dur   = (b-a+1)/fs;
        feat = [feat; table( ...
            c, a, b, dur, mx, a+imx-1, mn, a+imn-1, slope, ...
            'VariableNames',{'col','iStart','iEnd','dur','maxVal','iMax','minVal','iMin','slope'})]; %#ok<AGROW>
    end
    events{c} = struct('changePts',cps,'maskEdge',edgeMask(:,c),'indicesSpike',idxSpike{c});
    allFeat = [allFeat; feat]; %#ok<AGROW>
end
featTbl = allFeat;

end % function

% ---- 带掩码的 1D 双边滤波（非边缘区生效） -----------------------------
function y = bilateral1d_masked(x, mask, win, sigma_s, sigma_r, passes)
y = x;
half = max(1, floor(win/2));
n = numel(x);
idx = (1:n)';

for p = 1:passes
    y_new = y;
    for i = 1:n
        if ~mask(i)   % 边缘区：跳过（保持形状）
            continue
        end
        L = max(1, i-half); R = min(n, i+half);
        jj = (L:R)';
        % 空间权（时间距离）
        ds = abs(jj - i) / half;     % 归一化到 [0,1]
        ws = exp(-(ds.^2)/(2*sigma_s^2 + eps));
        % 幅值权（与当前值差异）
        dr = y(jj) - y(i);
        wr = exp(-(dr.^2)/(2*sigma_r^2 + eps));
        w  = ws .* wr;
        w  = w / max(eps, sum(w));
        y_new(i) = sum(w .* y(jj));
    end
    y = y_new;
end
end
