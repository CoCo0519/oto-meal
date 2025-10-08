function batch_export_wdenoise_denoise_only(inDir, outDir, opts)
% 仅导出 wdenoise 去噪结果（无 Approx），同时输出 .txt 与 .mat，二者波形严格一致
% Export ONLY wdenoise-denoised signal (no Approx), to both .txt and .mat with identical waveforms
%
% 用法示例 / Usage:
% batch_export_wdenoise_denoise_only('G:\Project-Swallow\hyx_data', ...
%     'G:\Project-Swallow\denoised_hyx_data', ...
%     'HeaderLines',1,'Recursive',true, ...
%     'Level',4,'Wavelet','sym8','Method','Bayes','Rule','Soft','Noise','LevelIndependent', ...
%     'DetrendMode','linear');
%
% 输入假设：首行（可选）表头 + 每行 6 列数值（若列数不同请告知）
% Input assumption: optional first header line + 6 numeric columns per row (tell me if different)

arguments
    inDir (1,:) char
    outDir (1,:) char
    opts.HeaderLines (1,1) double = 1              % 表头行数 | number of header lines
    opts.Recursive   (1,1) logical = true           % 是否递归扫描子目录 | recursive scan
    opts.SaveV73     (1,1) logical = false          % 保存 v7.3 MAT | save -v7.3

    % —— 严格 wdenoise 参数（默认较强：Level=4） ——
    % —— Strict wdenoise params (stronger default: Level=4) ——
    opts.Wavelet     (1,:) char   = 'sym8'
    opts.Level       (1,1) double = 4
    opts.Method      (1,:) char   = 'Bayes'
    opts.Rule        (1,:) char   = 'Soft'
    opts.Noise       (1,:) char   = 'LevelIndependent'

    % 去趋势（为增强去毛刺，默认 linear）
    % Detrend (linear by default to improve de-spiking)
    opts.DetrendMode (1,:) char   = 'linear'        % 'off'|'constant'|'linear'
end

% —— 确保输出目录存在 | ensure output folder exists
if ~exist(outDir,'dir'), mkdir(outDir); end

% —— 收集输入文件 | collect input files
if opts.Recursive
    files = dir(fullfile(inDir, '**', '*.txt'));
else
    files = dir(fullfile(inDir, '*.txt'));
end

% —— 与 App 对齐的边界延拓：对称 | DWT extension: symmetric (matches App)
oldmode = dwtmode('status','nodisplay'); %#ok<DWTMODE>
dwtmode('sym','nodisplay');              %#ok<DWTMODE>

for k = 1:numel(files)
    inPath = fullfile(files(k).folder, files(k).name);

    % —— 读取：表头 + 6 列数值 | read header + 6 numeric columns
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

    X = [C{:}];                      % N×6
    if isempty(X)
        warning('跳过（无法解析为6列）| Skip (cannot parse 6 cols): %s', inPath);
        continue
    end
    X = double(X);
    [n, m] = size(X);

    % —— 去趋势（linear 默认开启）| detrend (linear by default)
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

    % —— 严格 wdenoise 去噪 | strict wdenoise denoising
    den = zeros(n,m);                % 去噪结果 | denoised result
    for c = 1:m
        den(:,c) = wdenoise(X_use(:,c), opts.Level, ...
            'Wavelet',         opts.Wavelet, ...
            'DenoisingMethod', opts.Method, ...
            'ThresholdRule',   opts.Rule, ...
            'NoiseEstimate',   opts.Noise);
    end

    % —— 列名尝试解析（用于 .mat 说明，不影响 .txt 数值）| parse column names
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

    % —— 输出路径 | output paths
    relPath              = erase(inPath, [char(inDir) filesep]);
    [relFolder, base, ~] = fileparts(relPath);
    outFolder            = fullfile(outDir, relFolder);
    if ~exist(outFolder,'dir'), mkdir(outFolder); end
    outTxtDenoise        = fullfile(outFolder, [base '_denoise.txt']);
    outMat               = fullfile(outFolder, [base '_denoise.mat']);

    % —— 写 TXT（仅 denoise），精度使用 %.15g，保证与 .mat 一致 | write TXT (denoise only) with %.15g
    fid = fopen(outTxtDenoise,'w','n','UTF-8');
    desc = sprintf('# denoise = wdenoise(Level=%d, Wavelet=%s, Method=%s, Rule=%s, Noise=%s), detrend=%s', ...
                    opts.Level, opts.Wavelet, opts.Method, opts.Rule, opts.Noise, opts.DetrendMode);
    fprintf(fid, '%s\n', desc);
    for h = 1:numel(headers)
        if strlength(headers(h))>0, fprintf(fid,'%s\n', headers(h)); end
    end
    % 逐行逐列写出，固定格式与分隔，防止精度/区域设置差异
    % Write row-by-row with fixed format/delimiter to avoid precision/locale issues
    fmt = [repmat('%.15g\t', 1, m-1), '%.15g\n'];
    for r = 1:n
        fprintf(fid, fmt, den(r,:));
    end
    fclose(fid);

    % —— 写 MAT（保存与 .txt 同一份矩阵）| write MAT (save the exact same matrix as in .txt)
    data           = den;               %#ok<NASGU>  % 与 .txt 完全相同的矩阵 | exactly same as .txt
    data_raw       = X;                 %#ok<NASGU>
    data_used      = X_use;             %#ok<NASGU>  % 去趋势后用于去噪的输入 | detrended input to wdenoise
    names          = cellstr(names);    %#ok<NASGU>
    meta = struct();                    %#ok<NASGU>
    meta.source_path   = inPath;
    meta.header_lines  = opts.HeaderLines;
    meta.params        = rmfield(opts, {'Recursive','SaveV73'});

    if opts.SaveV73
        save(outMat, 'data','data_raw','data_used','names','meta','-v7.3');
    else
        save(outMat, 'data','data_raw','data_used','names','meta');
    end

    fprintf('OK: %s\n  TXT(denoise): %s\n  MAT: %s\n', inPath, outTxtDenoise, outMat);
end

% —— 恢复 DWT 模式 | restore DWT mode
dwtmode(oldmode,'nodisplay'); %#ok<DWTMODE>
end
