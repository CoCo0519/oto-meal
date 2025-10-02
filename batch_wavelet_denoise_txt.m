function batch_wavelet_denoise_txt(inDir, outDir, opts)
% 批量对"首行文本 + 6列数值"的 .txt 做小波去噪。       |
% Batch denoise ".txt" files with 1 text header line + 6 numeric columns. |

arguments
    inDir (1,:) char
    outDir (1,:) char
    opts.SignalColumns (1,:) double = 1:6   % 要去噪的列（默认全部6列）| Columns to denoise (default all 6)
    opts.HeaderLines   (1,1) double = 1     % 需要跳过的表头行数          | Number of header lines to skip
    opts.Suffix        (1,:) char = '_denoise'
    opts.Recursive     (1,1) logical = true
end

if ~exist(outDir,'dir'), mkdir(outDir); end
% —— 收集待处理的 .txt 文件 —— | Collect .txt files to process
if opts.Recursive
    files = dir(fullfile(inDir, '**', '*.txt'));   % 递归 ｜ recursive
else
    files = dir(fullfile(inDir, '*.txt'));         % 仅当前目录 ｜ non-recursive
end

for k = 1:numel(files)
    inPath = fullfile(files(k).folder, files(k).name);

    % —— 读取：先取表头，再读6列数值 —— | Read: grab header, then parse 6 numeric columns
    fid = fopen(inPath,'r','n','UTF-8');
    headers = strings(opts.HeaderLines,1);
    for h = 1:opts.HeaderLines
        tline = fgetl(fid);
        if ischar(tline), headers(h) = string(tline); end
    end
    C = textscan(fid, '%f%f%f%f%f%f', ...
        'Delimiter', {'\t',' ',',',';'}, 'MultipleDelimsAsOne', true);
    fclose(fid);
    X = [C{:}];                       % N×6
    if isempty(X)
        warning('跳过（无法解析为6列数值）：%s', inPath); % Skip if not numeric
        continue
    end
    X = double(X);
    Y = X;

    % —— 与 App 一致的小波去噪参数 —— | Denoising params identical to the App
    wav   = 'sym8';
    lvl   = 7;
    meth  = 'Bayes';
    rule  = 'Soft';
    noise = 'LevelIndependent';

    % 逐列去噪（只处理 SignalColumns 指定的列） | Denoise per selected column
    for c = opts.SignalColumns
        if c <= size(X,2)
            Y(:,c) = wdenoise(X(:,c), lvl, ...
                'Wavelet',         wav, ...
                'DenoisingMethod', meth, ...
                'ThresholdRule',   rule, ...
                'NoiseEstimate',   noise);
        end
    end

    % —— 生成输出路径并写回（保留原首行） —— | Build output path and write back (preserve header)
    relPath            = erase(inPath, [char(inDir) filesep]);
    [relFolder, base, ext] = fileparts(relPath);
    outFolder          = fullfile(outDir, relFolder);
    if ~exist(outFolder,'dir'), mkdir(outFolder); end
    outPath            = fullfile(outFolder, [base opts.Suffix ext]);

    fid = fopen(outPath,'w','n','UTF-8');
    for h = 1:numel(headers)
        if strlength(headers(h))>0, fprintf(fid,'%s\n', headers(h)); end
    end
    fclose(fid);
    writematrix(Y, outPath, 'Delimiter','\t', 'WriteMode','append');

    fprintf('Done: %s -> %s\n', inPath, outPath);
end
end
