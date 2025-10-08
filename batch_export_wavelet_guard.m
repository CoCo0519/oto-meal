function batch_export_wavelet_guard(inDir, outDir, opts)
% 批量：小波主导去毛刺 + 边缘保护 + cycle-spinning；导出 Approx TXT / MAT
% Batch: wavelet-first despike + edge guard + cycle-spinning; export Approx TXT / MAT
%
% 示例 | Example:
% batch_export_wavelet_guard('G:\Project-Swallow\hyx_data', ...
%   'G:\Project-Swallow\denoised_hyx_data', ...
%   'HeaderLines',1,'Recursive',true,'SaveV73',false, ...
%   'Fs',40,'Detrend','linear','Wavelet','sym8','Level',4, ...
%   'ThreshRule','Soft','EdgeMethod','deriv','EdgeQuantile',0.985, ...
%   'GuardRadius',0.10,'CycleSpin',16,'Kfine',1.2,'Kmid',0.9,'Kguard',0.3, ...
%   'ZeroFineOutsideEdge',true,'FineZeroLevels',[1 2], ...
%   'DoSgolay',true,'SgolayWinSec',0.06,'SgolayPoly',2, ...
%   'ApproxLevel',2);

arguments
    inDir (1,:) char
    outDir (1,:) char
    opts.HeaderLines (1,1) double = 1
    opts.Recursive   (1,1) logical = true
    opts.SaveV73     (1,1) logical = false

    % 采样与预处理 | sampling & preprocess
    opts.Fs          (1,1) double = 40
    opts.Detrend     (1,:) char   = 'linear'    % 'off'|'constant'|'linear'

    % 小波与阈值 | wavelet & threshold
    opts.Wavelet     (1,:) char   = 'sym8'
    opts.Level       (1,1) double = 4
    opts.ThreshRule  (1,:) char   = 'Soft'      % 'Soft'|'Hard'
    opts.EdgeMethod  (1,:) char   = 'deriv'     % 'deriv'|'changepts'
    opts.EdgeQuantile(1,1) double = 0.985
    opts.GuardRadius (1,1) double = 0.10       % seconds
    opts.CycleSpin   (1,1) double = 16
    opts.Kfine       (1,1) double = 1.2
    opts.Kmid        (1,1) double = 0.9
    opts.Kguard      (1,1) double = 0.3

    % 关键：细尺度"非边缘区"直接置零 | key: zero fine details outside edges
    opts.ZeroFineOutsideEdge (1,1) logical = true
    opts.FineZeroLevels      (1,:) double  = [1 2]   % 哪些细层置零

    % 轻度后置平滑（仅非边缘区）| mild post-smoothing (non-edge only)
    opts.DoSgolay    (1,1) logical = true
    opts.SgolayWinSec(1,1) double  = 0.06
    opts.SgolayPoly  (1,1) double  = 2

    % 导出近似层 | exported Approx level
    opts.ApproxLevel (1,1) double = 2
end

% 输出目录
if ~exist(outDir,'dir'), mkdir(outDir); end

% 输入文件
if opts.Recursive
    files = dir(fullfile(inDir, '**', '*.txt'));
else
    files = dir(fullfile(inDir, '*.txt'));
end

% DWT 边界：对齐 App
oldmode = dwtmode('status','nodisplay'); %#ok<DWTMODE>
dwtmode('sym','nodisplay');              %#ok<DWTMODE>

for k = 1:numel(files)
    inPath = fullfile(files(k).folder, files(k).name);

    % 读取
    fid = fopen(inPath,'r','n','UTF-8');
    if fid < 0, warning('无法打开 | Cannot open: %s', inPath); continue; end
    headers = strings(max(0,opts.HeaderLines),1);
    for h = 1:opts.HeaderLines
        tline = fgetl(fid); if ischar(tline), headers(h) = string(tline); end
    end
    C = textscan(fid, '%f%f%f%f%f%f', 'Delimiter', {'\t',' ',',',';'}, 'MultipleDelimsAsOne', true);
    fclose(fid);
    X = [C{:}];
    if isempty(X), warning('跳过（无法解析为6列）：%s', inPath); continue; end
    X = double(X); [n,m] = size(X);

    % 主处理
    [Y, info] = wavelet_despike_edge(X, opts.Fs, ...
        'Wavelet',opts.Wavelet, 'Level',opts.Level, 'ThreshRule',opts.ThreshRule, ...
        'EdgeMethod',opts.EdgeMethod, 'EdgeQuantile',opts.EdgeQuantile, ...
        'GuardRadius',round(opts.GuardRadius*opts.Fs), ...
        'CycleSpin',opts.CycleSpin, ...
        'Kfine',opts.Kfine,'Kmid',opts.Kmid,'Kguard',opts.Kguard, ...
        'ZeroFineOutsideEdge',opts.ZeroFineOutsideEdge,'FineZeroLevels',opts.FineZeroLevels, ...
        'DoSgolay',opts.DoSgolay,'SgolayWin',max(3,round(opts.SgolayWinSec*opts.Fs)),'SgolayPoly',opts.SgolayPoly, ...
        'Detrend',opts.Detrend);

    % 从清洗信号重构 Approx
    A = zeros(n,m);
    for c = 1:m
        [cvec, lvec] = wavedec(Y(:,c), opts.ApproxLevel, opts.Wavelet);
        A(:,c) = wrcoef('a', cvec, lvec, opts.Wavelet, opts.ApproxLevel);
    end

    % 列名
    names = repmat("", 1, m);
    if ~isempty(headers)
        last = strtrim(headers(end));
        if strlength(last)>0
            parts = regexp(last,'[\t,; ]+','split');
            parts = parts(~cellfun('isempty',parts));
            for i=1:min(numel(parts),m), names(i)=string(parts{i}); end
        end
    end
    for i=1:m, if strlength(names(i))==0, names(i)="col"+string(i); end, end

    % 输出路径
    relPath              = erase(inPath, [char(inDir) filesep]);
    [relFolder, base, ~] = fileparts(relPath);
    outFolder            = fullfile(outDir, relFolder);
    if ~exist(outFolder,'dir'), mkdir(outFolder); end
    outTxtApprox         = fullfile(outFolder, [base '_approx.txt']);
    outMat               = fullfile(outFolder, [base '_denoise.mat']);

    % 写 TXT（仅 Approx）
    fid = fopen(outTxtApprox,'w','n','UTF-8');
    desc = sprintf(['# content=Approx(A%d) wavelet-guard(' ...
        'W=%s,L=%d,Rule=%s,Edge=%s,q=%.3f,GR=%gs,CSpin=%d,Kf=%.2f,Km=%.2f,Kg=%.2f,' ...
        'ZeroFine=%d,FineLv=%s,Sgolay=%d,Win=%d,Poly=%d,Detrend=%s) | Fs=%.3f'], ...
        opts.ApproxLevel, opts.Wavelet, opts.Level, opts.ThreshRule, ...
        opts.EdgeMethod, opts.EdgeQuantile, opts.GuardRadius, opts.CycleSpin, ...
        opts.Kfine, opts.Kmid, opts.Kguard, ...
        opts.ZeroFineOutsideEdge, mat2str(opts.FineZeroLevels), opts.DoSgolay, ...
        max(3,round(opts.SgolayWinSec*opts.Fs)), opts.SgolayPoly, opts.Detrend, opts.Fs);
    fprintf(fid, '%s\n', desc);
    for h = 1:numel(headers)
        if strlength(headers(h))>0, fprintf(fid,'%s\n', headers(h)); end
    end
    fclose(fid);
    writematrix(A, outTxtApprox, 'Delimiter','\t', 'WriteMode','append');

    % 写 MAT
    approx   = A;                %#ok<NASGU>
    data     = Y;                %#ok<NASGU>
    data_raw = X;                %#ok<NASGU>
    names    = cellstr(names);   %#ok<NASGU>
    meta = struct();             %#ok<NASGU>
    meta.source_path  = inPath;
    meta.header_lines = opts.HeaderLines;
    meta.params       = opts;
    meta.guard_info   = info;

    if opts.SaveV73
        save(outMat, 'approx','data','data_raw','names','meta','-v7.3');
    else
        save(outMat, 'approx','data','data_raw','names','meta');
    end

    fprintf('OK: %s\n  TXT: %s\n  MAT: %s\n', inPath, outTxtApprox, outMat);
end

dwtmode(oldmode,'nodisplay'); %#ok<DWTMODE>
end

% ================== 主算法：小波守护 + 细层置零 + S-G 平滑 ==================
function [y, info] = wavelet_despike_edge(x, fs, varargin)
p = inputParser;
addParameter(p,'Wavelet','sym8');
addParameter(p,'Level',4);
addParameter(p,'ThreshRule','Soft');
addParameter(p,'EdgeMethod','deriv');
addParameter(p,'EdgeQuantile',0.985);
addParameter(p,'GuardRadius',round(0.10*fs));
addParameter(p,'CycleSpin',16);
addParameter(p,'Kfine',1.2);
addParameter(p,'Kmid',0.9);
addParameter(p,'Kguard',0.3);
addParameter(p,'ZeroFineOutsideEdge',true);
addParameter(p,'FineZeroLevels',[1 2]);
addParameter(p,'DoSgolay',true);
addParameter(p,'SgolayWin',max(3,round(0.06*fs)));
addParameter(p,'SgolayPoly',2);
addParameter(p,'Detrend','linear');
parse(p,varargin{:});
P = p.Results;

x = double(x);
[N,C] = size(x);

% 去趋势
switch lower(P.Detrend)
    case 'off'
    case 'constant', for c=1:C, x(:,c)=detrend(x(:,c),'constant'); end
    case 'linear',   for c=1:C, x(:,c)=detrend(x(:,c),'linear');   end
end

% 边缘掩码
maskEdge = false(N,C); changePts = cell(1,C);
for c=1:C
    xc = x(:,c);
    if strcmpi(P.EdgeMethod,'deriv')
        dx = gradient(xc);
        thr = quantile(abs(dx), P.EdgeQuantile);
        m = abs(dx) >= thr;
        rad = max(1, P.GuardRadius);
        m = movmax(m, [rad rad]);
        maskEdge(:,c) = m; changePts{c} = [];
    else
        try
            cp1 = findchangepts(xc,'Statistic','mean','MaxNumChanges',10,'MinThreshold',1);
            cp2 = findchangepts(xc,'Statistic','linear','MaxNumChanges',10,'MinThreshold',1);
            cp  = unique([cp1(:); cp2(:)]);
        catch, cp = [];
        end
        m = false(N,1); rad = max(1, P.GuardRadius);
        for kk=1:numel(cp), a=max(1,cp(kk)-rad); b=min(N,cp(kk)+rad); m(a:b)=true; end
        maskEdge(:,c)=m; changePts{c}=cp;
    end
end

% cycle-spinning
y_acc = zeros(N,C);
for sft = 0:(P.CycleSpin-1)
    xs = circshift(x, sft, 1);
    ms = circshift(maskEdge, sft, 1);

    ys = zeros(N,C);
    for c=1:C
        [cvec,lvec] = wavedec(xs(:,c), P.Level, P.Wavelet);
        for lev = 1:P.Level
            d = detcoef(cvec,lvec,lev); d = d(:); n_j = numel(d);
            sigma = median(abs(d))/0.6745; if sigma<eps, sigma=std(d); end
            if sigma<eps, sigma=1; end
            scaleK = (lev<=2)*P.Kfine + (lev>2)*P.Kmid;
            Tbase  = scaleK*sigma*sqrt(2*log(max(N,2)));

            % 时域边缘 -> 系数掩码
            blk = 2^lev; mlev=false(n_j,1);
            for k2=1:n_j
                a=(k2-1)*blk+1; b=min(N,k2*blk);
                if any(ms(a:b)), mlev(k2)=true; end
            end
            T = Tbase*ones(n_j,1);
            T(mlev) = T(mlev)*P.Kguard;  % 边缘内减阈值（保护）

            % **核心**：细层在"非边缘区"直接置零，杀掉锯齿
            if P.ZeroFineOutsideEdge && ismember(lev, P.FineZeroLevels)
                d(~mlev) = 0;
                % 边缘区保留温和软阈
                if any(mlev)
                    d(mlev) = sign(d(mlev)) .* max(abs(d(mlev)) - T(mlev), 0);
                end
            else
                % 常规阈值（软/硬）
                if strcmpi(P.ThreshRule,'hard')
                    d = d .* (abs(d) >= T);
                else
                    d = sign(d) .* max(abs(d) - T, 0);
                end
            end

            cvec = inject_detail(cvec,lvec,lev,d);
        end
        ys(:,c) = waverec(cvec,lvec,P.Wavelet);
    end
    y_acc = y_acc + circshift(ys, -sft, 1);
end
y = y_acc / P.CycleSpin;

% 轻度 S-G 平滑（仅非边缘区，避免事件被抹）
if P.DoSgolay
    win = max(3, P.SgolayWin + mod(P.SgolayWin+1,2)); % 奇数窗
    for c=1:C
        ne = ~maskEdge(:,c);
        if nnz(ne) > win
            tmp = y(:,c);
            tmp(ne) = sgolayfilt(tmp(ne), P.SgolayPoly, win);
            y(:,c) = tmp;
        end
    end
end

info = struct('maskEdge',maskEdge,'changePts',{changePts},'params',P);
end

% ---------- detail 写回：与 wavedec 的 C/L 对齐 ----------
function cvec2 = inject_detail(cvec, lvec, lev, dnew)
% cvec: [cA_L, cD_L, cD_{L-1}, ..., cD1]
% lvec: [len(cA_L), len(cD_L), len(cD_{L-1}), ..., len(cD1)]
cvec2 = cvec;
Nlev = numel(lvec) - 2;
if lev < 1 || lev > Nlev, error('inject_detail: level out of range'); end
idxL = Nlev - lev + 2;        % L(2)=cD_L, ... , L(end)=cD1
lenD = lvec(idxL);
if numel(dnew) ~= lenD
    error('inject_detail: length mismatch (got %d, expect %d).', numel(dnew), lenD);
end
start = 1 + lvec(1);          % 跳过 cA_L
if idxL > 2, start = start + sum(lvec(2:idxL-1)); end
stop = start + lenD - 1;
cvec2(start:stop) = dnew(:);
end
