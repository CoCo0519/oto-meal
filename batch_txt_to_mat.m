function batch_txt_to_mat(inDir, outDir, opts)
% 批量将 6 列 .txt 转换为 .mat；固定变量名：g, ir, rd, imu_x, imu_y, imu_z
% Batch-convert 6-column .txt files to .mat; fixed variable names: g, ir, rd, imu_x, imu_y, imu_z
%
% 用法 / Usage:
%   batch_txt_to_mat('G:\Project-Swallow\hyx_data', 'G:\Project-Swallow\hyx_data\mat')
%   batch_txt_to_mat('G:\in', 'G:\out', 'HeaderLines',1, 'Recursive',true, 'Fs', 100)  % 可选 Fs
%
% 说明 / Notes:
% - 假设 .txt：首行表头 + 每行 6 列数值（与您示例一致）
% - Assumes .txt: one header line + 6 numeric columns (matches your sample)

arguments
    inDir (1,:) char
    outDir (1,:) char
    opts.Recursive   (1,1) logical = true
    opts.HeaderLines (1,1) double  = 1
    opts.Fs          (1,1) double  = NaN   % 用 NaN 作为"未提供"的占位 | NaN means "not provided"
end

FORCED_NAMES = {'g','ir','rd','imu_x','imu_y','imu_z'};

if ~exist(outDir,'dir'), mkdir(outDir); end

% 收集 .txt 文件 | Collect .txt files
if opts.Recursive
    files = dir(fullfile(inDir,'**','*.txt'));
else
    files = dir(fullfile(inDir,'*.txt'));
end

for k = 1:numel(files)
    inPath = fullfile(files(k).folder, files(k).name);

    % 读取：跳过表头，解析 6 列 | Read: skip headers, parse 6 numeric cols
    fid = fopen(inPath,'r','n','UTF-8');
    if fid < 0
        warning('Cannot open: %s', inPath); continue;
    end
    for h = 1:opts.HeaderLines
        fgetl(fid); % skip header
    end
    C = textscan(fid, '%f%f%f%f%f%f', ...
        'Delimiter', {'\t',' ',',',';'}, 'MultipleDelimsAsOne', true);
    fclose(fid);

    X = [C{:}];   % N×6
    if isempty(X) || size(X,2)~=6
        warning('Skip (not 6 numeric cols): %s', inPath);
        continue;
    end
    X = double(X);

    % 拆列并命名 | Split columns and name
    g      = X(:,1); %#ok<NASGU>
    ir     = X(:,2); %#ok<NASGU>
    rd     = X(:,3); %#ok<NASGU>
    imu_x  = X(:,4); %#ok<NASGU>
    imu_y  = X(:,5); %#ok<NASGU>
    imu_z  = X(:,6); %#ok<NASGU>
    data6  = X;      %#ok<NASGU>
    names  = FORCED_NAMES; %#ok<NASGU>

    % 输出路径 | Output path
    rel    = erase(inPath, [char(inDir) filesep]);
    [relFolder, base, ~] = fileparts(rel);
    outFolder = fullfile(outDir, relFolder);
    if ~exist(outFolder,'dir'), mkdir(outFolder); end
    outMat = fullfile(outFolder, [base '.mat']);

    % 可选保存 Fs | Optionally save Fs
    if isnan(opts.Fs)
        save(outMat, 'g','ir','rd','imu_x','imu_y','imu_z','data6','names');
    else
        Fs = opts.Fs; %#ok<NASGU>
        save(outMat, 'g','ir','rd','imu_x','imu_y','imu_z','data6','names','Fs');
    end

    fprintf('OK: %s -> %s\n', inPath, outMat);
end
end
