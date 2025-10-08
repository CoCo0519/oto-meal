function batch_wavelet_denoise_txt(inDir, outDir, opts)
% 批量对"首行文本 + 6列数值"的 .txt 做小波去噪，并导出 .txt + .mat（含近似分量 A7）。
% Batch denoise ".txt" files (1 header line + 6 numeric cols), export .txt + .mat
% with A7 approximation included.
%
% MAT 中变量：
%   data      : N×M 去噪结果（wdenoise）
%   approx    : N×M 去噪结果再次分解后在 Level=7 的"近似分量 A7"（最平滑，Denoiser 面板中的
%               "Approximation"即此类量；用于与你截图中的白色虚线对齐）
%   data_raw  : N×M 原始数据
%   names     : 1×M 列名（尽力从表头解析，否则 "col1"...）
%   meta      : 结构体（参数、路径、处理列等；用于检索/复现）
%
% 依赖：Wavelet Toolbox
%
% 示例：
%   batch_wavelet_denoise_txt('G:\Project-Swallow\hyx_data', ...
%     'G:\Project-Swallow\denoised_hyx_data', ...
%     'SignalColumns',1:6,'HeaderLines',1,'Recursive',true,'SaveV73',false);

arguments
    inDir (1,:) char
    outDir (1,:) char
    opts.SignalColumns (1,:) double = 1:6
    opts.HeaderLines   (1,1) double = 1
    opts.Suffix        (1,:) char   = '_denoise'
    opts.Recursive     (1,1) logical = true
    opts.SaveV73       (1,1) logical = false
end

if ~exist(outDir,'dir'), mkdir(outDir); end

% 收集文件
if opts.Recursive
    files = dir(fullfile(inDir, '**', '*.txt'));
else
    files = dir(fullfile(inDir, '*.txt'));
end

% 与 Denoiser UI 一致的参数
wav   = 'sym8';
lvl   = 7;
meth  = 'Bayes';
rule  = 'Soft';
noise = 'LevelIndependent';

% 边界延拓（App 默认为 symmetric）
oldmode = dwtmode('status','nodisplay'); %#ok<DWTMODE>
dwtmode('sym','nodisplay');              %#ok<DWTMODE>

for k = 1:numel(files)
    inPath = fullfile(files(k).folder, files(k).name);

    % 读表头 + 6 列数值
    fid = fopen(inPath,'r','n','UTF-8');
    if fid < 0, warning('无法打开：%s', inPath); continue; end
    headers = strings(max(0,opts.HeaderLines),1);
    for h = 1:opts.HeaderLines
        tline = fgetl(fid);
        if ischar(tline), headers(h) = string(tline); end
    end
    C = textscan(fid, '%f%f%f%f%f%f', ...
        'Delimiter', {'\t',' ',',',';'}, 'MultipleDelimsAsOne', true);
    fclose(fid);
    X = [C{:}];
    if isempty(X)
        warning('跳过（无法解析为6列数值）：%s', inPath);
        continue
    end
    X = double(X);
    Y = X;  % 去噪结果
    A = X;  % 近似分量（A7），后面填充

    % 逐列去噪
    for c = opts.SignalColumns
        if c <= size(X,2)
            yc = wdenoise(X(:,c), lvl, ...
                'Wavelet',         wav, ...
                'DenoisingMethod', meth, ...
                'ThresholdRule',   rule, ...
                'NoiseEstimate',   noise);
            Y(:,c) = yc;

            % === 额外：从"去噪后的 yc"再做一次 DWT 分解，提取 Level=7 的近似分量（A7） ===
            % 与 Denoiser 面板 "Show Approximation" 的平滑趋势量对齐
            [cvec, lvec] = wavedec(yc, lvl, wav);
            A(:,c) = wrcoef('a', cvec, lvec, wav, lvl);  % A7
        end
    end

    % 解析列名
    names = repmat("", 1, size(X,2));
    if ~isempty(headers)
        last = strtrim(headers(end));
        if strlength(last) > 0
            parts = regexp(last, '[\t,; ]+', 'split');
            parts = parts(~cellfun('isempty',parts));
            for i = 1:min(numel(parts), size(X,2))
                names(i) = string(parts{i});
            end
        end
    end
    for i = 1:size(X,2)
        if strlength(names(i)) == 0, names(i) = "col"+string(i); end
    end

    % 输出路径
    relPath              = erase(inPath, [char(inDir) filesep]);
    [relFolder, base, ~] = fileparts(relPath);
    outFolder            = fullfile(outDir, relFolder);
    if ~exist(outFolder,'dir'), mkdir(outFolder); end
    outTxt               = fullfile(outFolder, [base opts.Suffix '.txt']);
    outMat               = fullfile(outFolder, [base opts.Suffix '.mat']);

    % 写 .txt（保留原表头）
    fid = fopen(outTxt,'w','n','UTF-8');
    for h = 1:numel(headers)
        if strlength(headers(h))>0, fprintf(fid,'%s\n', headers(h)); end
    end
    fclose(fid);
    writematrix(Y, outTxt, 'Delimiter','\t', 'WriteMode','append');

    % 写 .mat（含 approx）
    data      = Y;               %#ok<NASGU>
    approx    = A;               %#ok<NASGU>
    data_raw  = X;               %#ok<NASGU>
    names     = cellstr(names);  %#ok<NASGU>
    meta      = struct();        %#ok<NASGU>
    meta.source_path     = inPath;
    meta.signal_columns  = opts.SignalColumns;
    meta.header_lines    = opts.HeaderLines;
    meta.params.wavelet  = wav;
    meta.params.level    = lvl;
    meta.params.method   = meth;
    meta.params.rule     = rule;
    meta.params.noise    = noise;
    if opts.SaveV73
        save(outMat, 'data','approx','data_raw','names','meta','-v7.3');
    else
        save(outMat, 'data','approx','data_raw','names','meta');
    end

    fprintf('Done: %s ->\n      %s\n      %s\n', inPath, outTxt, outMat);
end

% 还原边界延拓
dwtmode(oldmode,'nodisplay'); %#ok<DWTMODE>
end
