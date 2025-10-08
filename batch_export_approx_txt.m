function batch_export_approx_txt(inDir, outDir, opts)
% 批量：小波降噪 (wdenoise) + 直接重构 A(Approx) 导出 TXT / MAT
% Batch: wavelet denoise (wdenoise) + reconstruct Approx A to export TXT / MAT
%
% 用法示例 | Example:
%   batch_export_approx_txt('G:\Project-Swallow\hyx_data', ...
%       'G:\Project-Swallow\denoised_hyx_data', ...
%       'HeaderLines',1,'Recursive',true,'SaveV73',false, ...
%       'DetrendMode','linear','Wavelet','sym8','DenoiseLevel',2,'ApproxLevel',2);
%
% 假设输入 .txt ：首行表头 + 每行 6 列数值（若列数不同可自行修改读取格式）
% Assumes input .txt: first header + 6 numeric columns (change the reader if needed)

arguments
    inDir (1,:) char
    outDir (1,:) char
    opts.HeaderLines (1,1) double = 1                 % 首行表头行数 | header lines
    opts.Recursive   (1,1) logical = true              % 递归扫描子目录 | recursive scan
    opts.SaveV73     (1,1) logical = false             % MAT v7.3
    opts.DetrendMode (1,:) char   = 'linear'           % 'off'|'constant'|'linear'
    opts.Wavelet     (1,:) char   = 'sym8'             % 小波基 | wavelet
    opts.DenoiseLevel(1,1) double = 2                  % 降噪分解层数 | level for wdenoise
    opts.ApproxLevel (1,1) double = 2                  % 近似分量层数 | level for Approx A
    opts.Method      (1,:) char   = 'Bayes'            % 去噪方法 | denoising method
    opts.Rule        (1,:) char   = 'Soft'             % 阈值规则 | threshold rule
    opts.Noise       (1,:) char   = 'LevelIndependent' % 噪声估计 | noise estimate
end

% 确保输出目录存在 | ensure output folder exists
if ~exist(outDir,'dir'), mkdir(outDir); end

% 收集输入文件 | collect input files
if opts.Recursive
    files = dir(fullfile(inDir, '**', '*.txt'));
else
    files = dir(fullfile(inDir, '*.txt'));
end

% DWT 边界延拓（与 App 一致：symmetric）| DWT extension (App default: symmetric)
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

    X = [C{:}]; % N×6
    if isempty(X)
        warning('跳过（无法解析为6列）| Skip (cannot parse 6 cols): %s', inPath);
        continue
    end
    X = double(X);
    [n, m] = size(X);

    % —— 去趋势（可选）| detrend (optional)
    X_det = X;
    switch lower(opts.DetrendMode)
        case 'off'
        case 'constant'
            for c=1:m, X_det(:,c) = detrend(X(:,c),'constant'); end
        case 'linear'
            for c=1:m, X_det(:,c) = detrend(X(:,c),'linear');   end
        otherwise
            error('DetrendMode must be off|constant|linear');
    end

    % —— wdenoise —— 
    Y = zeros(n,m); % 去噪结果 | denoised
    for c = 1:m
        Y(:,c) = wdenoise(X_det(:,c), opts.DenoiseLevel, ...
            'Wavelet',         opts.Wavelet, ...
            'DenoisingMethod', opts.Method, ...
            'ThresholdRule',   opts.Rule, ...
            'NoiseEstimate',   opts.Noise);
    end

    % —— 由去噪信号重构 Approx A(level) —— 
    % —— Reconstruct Approx A(level) from denoised signal ——
    A = zeros(n,m);
    for c = 1:m
        [cvec, lvec] = wavedec(Y(:,c), opts.ApproxLevel, opts.Wavelet);
        A(:,c) = wrcoef('a', cvec, lvec, opts.Wavelet, opts.ApproxLevel);
    end

    % —— 列名尝试解析 —— 
    % —— Parse column names ——
    names = repmat("", 1, m);
    if ~isempty(headers)
        last = strtrim(headers(end));
        if strlength(last) > 0
            parts = regexp(last, '[\t,; ]+', 'split');
            parts = parts(~cellfun('isempty',parts));
            for i = 1:min(numel(parts), m), names(i) = string(parts{i}); end
        end
    end
    for i = 1:m, if strlength(names(i))==0, names(i) = "col"+string(i); end, end

    % —— 输出路径 —— 
    relPath              = erase(inPath, [char(inDir) filesep]);
    [relFolder, base, ~] = fileparts(relPath);
    outFolder            = fullfile(outDir, relFolder);
    if ~exist(outFolder,'dir'), mkdir(outFolder); end
    outTxtApprox         = fullfile(outFolder, [base '_approx.txt']);
    outMat               = fullfile(outFolder, [base '_denoise.mat']);

    % —— 写 TXT（仅 Approx）—— 
    fid = fopen(outTxtApprox,'w','n','UTF-8');
    desc = sprintf('# content=Approximation(A%d) of wdenoise(%s,%s,%s,%s) | detrend=%s', ...
                   opts.ApproxLevel, opts.Wavelet, opts.Method, opts.Rule, opts.Noise, opts.DetrendMode);
    fprintf(fid, '%s\n', desc);
    for h = 1:numel(headers)
        if strlength(headers(h))>0, fprintf(fid,'%s\n', headers(h)); end
    end
    fclose(fid);
    writematrix(A, outTxtApprox, 'Delimiter','\t', 'WriteMode','append');

    % —— 写 MAT（含更多变量）—— 
    approx         = A;               %#ok<NASGU>
    data           = Y;               %#ok<NASGU>
    data_raw       = X;               %#ok<NASGU>
    data_detrended = X_det;           %#ok<NASGU>
    names          = cellstr(names);  %#ok<NASGU>
    meta = struct();                  %#ok<NASGU>
    meta.source_path   = inPath;
    meta.header_lines  = opts.HeaderLines;
    meta.params = opts;
    if opts.SaveV73
        save(outMat, 'approx','data','data_raw','data_detrended','names','meta','-v7.3');
    else
        save(outMat, 'approx','data','data_raw','data_detrended','names','meta');
    end

    fprintf('OK: %s\n  TXT: %s\n  MAT: %s\n', inPath, outTxtApprox, outMat);
end

% 恢复 DWT 模式 | restore DWT mode
dwtmode(oldmode,'nodisplay'); %#ok<DWTMODE>
end
