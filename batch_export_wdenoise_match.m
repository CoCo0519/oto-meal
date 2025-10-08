function batch_export_wdenoise_match(inDir, outDir, opts)
% 批量：wdenoise 严格配置 + 去趋势（默认 linear）→ 导出 denoise / approx
% Batch: strict wdenoise + detrend (default linear) → export denoise / approx
%
% 用法 / Usage:
% batch_export_wdenoise_match('G:\Project-Swallow\hyx_data', ...
%     'G:\Project-Swallow\denoised_hyx_data', ...
%     'HeaderLines',1,'Recursive',true);
%
% 读取假设：首行表头 + 每行 6 列数值（若不同请告知）
% Read assumption: 1 header line + 6 numeric columns (tell me if different)

arguments
    inDir (1,:) char
    outDir (1,:) char
    opts.HeaderLines (1,1) double = 1            % 表头行数 | header lines
    opts.Recursive   (1,1) logical = true         % 递归扫描 | recursive
    opts.SaveV73     (1,1) logical = false        % 保存 v7.3 | save v7.3

    % —— 严格的 wdenoise 参数（默认更强：Level=4）——
    % —— Strict wdenoise params (stronger default: Level=4) ——
    opts.Wavelet     (1,:) char   = 'sym8'
    opts.Level       (1,1) double = 4
    opts.Method      (1,:) char   = 'Bayes'
    opts.Rule        (1,:) char   = 'Soft'
    opts.Noise       (1,:) char   = 'LevelIndependent'

    % 去趋势：开启（linear）以抑制基线漂移与尖刺
    % Detrend: on (linear) to suppress baseline drift & spikes
    opts.DetrendMode (1,:) char   = 'linear'      % 'off'|'constant'|'linear'

    % Approx 的层级：默认 NaN，运行时用 Level 替代
    % Approx level: default NaN, replaced by Level at runtime
    opts.ApproxLevel (1,1) double = NaN
end

% —— 若 ApproxLevel 未给出，则用 Level —— 
% —— If ApproxLevel not provided, use Level ——
if isnan(opts.ApproxLevel)
    opts.ApproxLevel = opts.Level;
end

% 输出目录
% Output folder
if ~exist(outDir,'dir'), mkdir(outDir); end

% 收集输入文件
% Collect input files
if opts.Recursive
    files = dir(fullfile(inDir, '**', '*.txt'));
else
    files = dir(fullfile(inDir, '*.txt'));
end

% 与 App 对齐：对称延拓
% Match App: symmetric extension
oldmode = dwtmode('status','nodisplay'); %#ok<DWTMODE>
dwtmode('sym','nodisplay');              %#ok<DWTMODE>

for k = 1:numel(files)
    inPath = fullfile(files(k).folder, files(k).name);

    % —— 读取：表头 + 6 列数值 ——
    % —— Read: header + 6 numeric columns ——
    fid = fopen(inPath,'r','n','UTF-8');
    if fid < 0
        warning('无法打开 | Cannot open: %s', inPath);
        continue
    end
    headers = strings(max(0,opts.HeaderLines),1);
    for h = 1:opts.HeaderLines
        tline = fgetl(fid);
        if ischar(tline), headers(h) = string(tline); end
    end
    C = textscan(fid, '%f%f%f%f%f%f', ...
        'Delimiter', {'\t',' ',',',';'}, 'MultipleDelimsAsOne', true);
    fclose(fid);

    X = [C{:}];                                % N×6
    if isempty(X)
        warning('跳过（无法解析为6列）| Skip (cannot parse 6 cols): %s', inPath);
        continue
    end
    X = double(X);
    [n, m] = size(X);

    % —— 去趋势（默认 linear）——
    % —— Detrend (default linear) ——
    X_use = X;
    switch lower(opts.DetrendMode)
        case 'off'
        case 'constant'
            for c=1:m, X_use(:,c) = detrend(X(:,c),'constant'); end
        case 'linear'
            for c=1:m, X_use(:,c) = detrend(X(:,c),'linear');   end
        otherwise
            error('DetrendMode must be off|constant|linear');
    end

    % —— 严格 wdenoise（Level=opts.Level）——
    % —— Strict wdenoise (Level=opts.Level) ——
    Y = zeros(n,m);                            % 去噪结果 / denoised
    for c = 1:m
        Y(:,c) = wdenoise(X_use(:,c), opts.Level, ...
            'Wavelet',         opts.Wavelet, ...
            'DenoisingMethod', opts.Method, ...
            'ThresholdRule',   opts.Rule, ...
            'NoiseEstimate',   opts.Noise);
    end

    % —— 由"去噪后的信号"重构 Approx（与 App 的虚线一致）——
    % —— Reconstruct Approx from *denoised* signal (same as App dashed) ——
    A = zeros(n,m);
    for c = 1:m
        [cvec, lvec] = wavedec(Y(:,c), opts.ApproxLevel, opts.Wavelet);
        A(:,c) = wrcoef('a', cvec, lvec, opts.Wavelet, opts.ApproxLevel);
    end

    % 尝试解析列名
    % Parse column names
    names = repmat("", 1, m);
    if ~isempty(headers)
        last = strtrim(headers(end));
        if strlength(last) > 0
            parts = regexp(last, '[\t,; ]+', 'split');
            parts = parts(~cellfun('isempty',parts));
            for i = 1:min(numel(parts), m), names(i) = string(parts{i}); end
        end
    end
    for i = 1:m
        if strlength(names(i)) == 0, names(i) = "col"+string(i); end
    end

    % 输出路径
    % Output paths
    relPath              = erase(inPath, [char(inDir) filesep]);
    [relFolder, base, ~] = fileparts(relPath);
    outFolder            = fullfile(outDir, relFolder);
    if ~exist(outFolder,'dir'), mkdir(outFolder); end

    outTxtApprox         = fullfile(outFolder, [base '_approx.txt']);
    outTxtDenoise        = fullfile(outFolder, [base '_denoise.txt']);
    outMat               = fullfile(outFolder, [base '_denoise.mat']);

    % 写 TXT（Approx）
    % Write TXT (Approx)
    fid = fopen(outTxtApprox,'w','n','UTF-8');
    descA = sprintf('# approx = A%d after wdenoise(Level=%d, Wavelet=%s, Method=%s, Rule=%s, Noise=%s), detrend=%s', ...
                     opts.ApproxLevel, opts.Level, opts.Wavelet, opts.Method, opts.Rule, opts.Noise, opts.DetrendMode);
    fprintf(fid, '%s\n', descA);
    for h = 1:numel(headers)
        if strlength(headers(h))>0, fprintf(fid,'%s\n', headers(h)); end
    end
    fclose(fid);
    writematrix(A, outTxtApprox, 'Delimiter','\t', 'WriteMode','append');

    % 写 TXT（Denoise）
    % Write TXT (Denoise)
    fid = fopen(outTxtDenoise,'w','n','UTF-8');
    descD = sprintf('# denoise = wdenoise(Level=%d, Wavelet=%s, Method=%s, Rule=%s, Noise=%s), detrend=%s', ...
                     opts.Level, opts.Wavelet, opts.Method, opts.Rule, opts.Noise, opts.DetrendMode);
    fprintf(fid, '%s\n', descD);
    for h = 1:numel(headers)
        if strlength(headers(h))>0, fprintf(fid,'%s\n', headers(h)); end
    end
    fclose(fid);
    writematrix(Y, outTxtDenoise, 'Delimiter','\t', 'WriteMode','append');

    % 写 MAT（便于核检）
    % Write MAT (for inspection)
    approx         = A;               %#ok<NASGU>
    data           = Y;               %#ok<NASGU>
    data_raw       = X;               %#ok<NASGU>
    data_used      = X_use;           %#ok<NASGU>
    names          = cellstr(names);  %#ok<NASGU>
    meta = struct();                  %#ok<NASGU>
    meta.source_path   = inPath;
    meta.header_lines  = opts.HeaderLines;
    meta.params        = rmfield(opts, {'Recursive','SaveV73'});
    if opts.SaveV73
        save(outMat, 'approx','data','data_raw','data_used','names','meta','-v7.3');
    else
        save(outMat, 'approx','data','data_raw','data_used','names','meta');
    end

    fprintf('OK: %s\n  TXT(approx): %s\n  TXT(denoise): %s\n  MAT: %s\n', ...
            inPath, outTxtApprox, outTxtDenoise, outMat);
end

% 恢复 DWT 模式
% Restore DWT mode
dwtmode(oldmode,'nodisplay'); %#ok<DWTMODE>
end
