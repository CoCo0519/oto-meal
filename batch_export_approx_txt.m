function batch_export_approx_txt(inDir, outDir, opts)
% 批量导出"小波近似 A7（Approximation）"为 TXT，并保存 MAT（approx/data/data_raw/names/meta）
%
% 假设输入 .txt 为：首行表头（中文也可） + 每行 6 列数值（PPG_green, IR, Red, IMU X/Y/Z）
% 若你的列数并非 6，请告知，我可给你"自适应列数"的版本。
%
% 导出的文件：
%   1) <name>_approx.txt : 仅 A7 近似（所有列）
%   2) <name>_denoise.mat: 变量包括
%       - approx   : N×6  的 A7 近似（基于去噪结果再分解重构）
%       - data     : N×6  的去噪结果（wdenoise）
%       - data_raw : N×6  的原始数据
%       - names    : 1×6  列名（尽力从表头解析，不足则 col1..col6）
%       - meta     : 结构体（参数、路径、处理列等）
%
% 小波参数（与 Signal Denoiser UI 一致）：
%   Wavelet='sym8', Level=7, DenoisingMethod='Bayes',
%   ThresholdRule='Soft', NoiseEstimate='LevelIndependent'
%
% 用法示例：
%   batch_export_approx_txt('G:\Project-Swallow\hyx_data', ...
%       'G:\Project-Swallow\denoised_hyx_data', ...
%       'HeaderLines',1, 'Recursive',true, 'SaveV73',false);
%
% 依赖：Wavelet Toolbox

arguments
    inDir (1,:) char
    outDir (1,:) char
    opts.HeaderLines (1,1) double = 1
    opts.Recursive   (1,1) logical = true
    opts.SaveV73     (1,1) logical = false
end

% 确保输出目录存在
if ~exist(outDir,'dir'), mkdir(outDir); end

% 收集输入文件
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

% DWT 边界延拓（App 默认为 symmetric）
oldmode = dwtmode('status','nodisplay'); %#ok<DWTMODE>
dwtmode('sym','nodisplay');              %#ok<DWTMODE>

for k = 1:numel(files)
    inPath = fullfile(files(k).folder, files(k).name);

    % —— 读取：先表头，再 6 列数值（PPG/IMU）——
    fid = fopen(inPath,'r','n','UTF-8');
    if fid < 0
        warning('无法打开：%s', inPath);
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
        warning('跳过（无法解析为6列数值）：%s', inPath);
        continue
    end
    X = double(X);
    n = size(X,1); m = size(X,2);

    % —— 去噪 + 近似（对所有列）——
    Y = zeros(n,m);     % 去噪结果
    A = zeros(n,m);     % A7 近似
    for c = 1:m
        yc = wdenoise(X(:,c), lvl, ...
            'Wavelet',         wav, ...
            'DenoisingMethod', meth, ...
            'ThresholdRule',   rule, ...
            'NoiseEstimate',   noise);
        Y(:,c) = yc;

        % 从去噪后的 yc 计算 A7（Approximation）
        [cvec, lvec] = wavedec(yc, lvl, wav);
        A(:,c) = wrcoef('a', cvec, lvec, wav, lvl);
    end

    % —— 解析列名（尽力从最后一行表头）——
    names = repmat("", 1, m);
    if ~isempty(headers)
        last = strtrim(headers(end));
        if strlength(last) > 0
            parts = regexp(last, '[\t,; ]+', 'split');
            parts = parts(~cellfun('isempty',parts));
            for i = 1:min(numel(parts), m)
                names(i) = string(parts{i});
            end
        end
    end
    for i = 1:m
        if strlength(names(i)) == 0, names(i) = "col"+string(i); end
    end

    % —— 输出路径 —— 
    relPath              = erase(inPath, [char(inDir) filesep]);
    [relFolder, base, ~] = fileparts(relPath);
    outFolder            = fullfile(outDir, relFolder);
    if ~exist(outFolder,'dir'), mkdir(outFolder); end
    outTxtApprox         = fullfile(outFolder, [base '_approx.txt']);
    outMat               = fullfile(outFolder, [base '_denoise.mat']);

    % —— 写 TXT：仅写 A7（Approximation）——
    fid = fopen(outTxtApprox,'w','n','UTF-8');
    % 额外写一行描述，保留原表头（便于回溯）
    desc = sprintf('# content=Approximation(A%d) of wdenoise(sym8,Bayes,Soft,LevelIndependent)', lvl);
    fprintf(fid, '%s\n', desc);
    for h = 1:numel(headers)
        if strlength(headers(h))>0, fprintf(fid,'%s\n', headers(h)); end
    end
    fclose(fid);
    writematrix(A, outTxtApprox, 'Delimiter','\t', 'WriteMode','append');

    % —— 写 MAT：approx + data + data_raw + names + meta —— 
    approx    = A;               %#ok<NASGU>
    data      = Y;               %#ok<NASGU>
    data_raw  = X;               %#ok<NASGU>
    names     = cellstr(names);  %#ok<NASGU>
    meta      = struct();        %#ok<NASGU>
    meta.source_path     = inPath;
    meta.header_lines    = opts.HeaderLines;
    meta.params.wavelet  = wav;
    meta.params.level    = lvl;
    meta.params.method   = meth;
    meta.params.rule     = rule;
    meta.params.noise    = noise;

    if opts.SaveV73
        save(outMat, 'approx','data','data_raw','names','meta', '-v7.3');
    else
        save(outMat, 'approx','data','data_raw','names','meta');
    end

    fprintf('OK: %s ->\n    TXT: %s\n    MAT: %s\n', inPath, outTxtApprox, outMat);
end

% 恢复 DWT 模式
dwtmode(oldmode,'nodisplay'); %#ok<DWTMODE>
end
