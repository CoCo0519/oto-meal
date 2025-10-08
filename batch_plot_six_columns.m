% batch_plot_six_columns.m
% 批量识别指定文件夹中“与示例相同格式”的文本数据，逐文件生成“六列六图同页”的图像。
% Batch-detect TXT files in a folder (same format as the example), and for each file,
% draw 6 subplots (one per column) on a single page and save the figure.
%
% 使用说明 Usage:
%   1) 直接运行本脚本；若未设置 folder_path，将弹出选择文件夹。
%   2) 默认只匹配 *_approx.txt；如果找不到，则退化为 *.txt。
%   3) 若你在工作区设置 Fs（采样率，例如 Fs=100），脚本会用秒做横轴；否则用样本索引。
%
% 兼容性：需 MATLAB R2019b+（使用 tiledlayout / exportgraphics）；更老版本可改用 subplot/saveas。
%
% 作者注：示例文件首部可能含两行非数值头信息（如说明行与列名行），脚本会自动识别需要跳过的行数。

clear; clc;

% ============ 用户可选：手动指定目录（把下一行改成你的路径） ============
% folder_path = 'C:\your\path\to\folder';

% 若未设置路径，则弹窗选择
if ~exist('folder_path','var') || isempty(folder_path) || ~isfolder(folder_path)
    folder_path = uigetdir(pwd, '请选择包含六列TXT的文件夹 / Choose folder with 6-column TXTs');
    if folder_path==0
        error('未选择文件夹 / No folder selected.');
    end
end
fprintf('扫描目录：%s\n', folder_path);

% 匹配模式：优先 *_approx.txt，其次 *.txt
patterns = {'*_approx.txt','*.txt'};

file_list = [];
for ip = 1:numel(patterns)
    S = dir(fullfile(folder_path, patterns{ip}));
    file_list = [file_list; S]; %#ok<AGROW>
    if ~isempty(S), break; end % 找到 *_approx.txt 即停止；否则继续尝试 *.txt
end
% 去重（同名只保留一个）
if ~isempty(file_list)
    [~, ia] = unique({file_list.name}, 'stable');
    file_list = file_list(ia);
end

if isempty(file_list)
    error('未在该目录下找到符合模式的TXT文件。请检查文件名或更换目录。');
end

% 输出目录
out_dir = fullfile(folder_path, 'fig_out');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

% 若工作区已有 Fs，则使用秒作为时间轴；否则用样本索引
use_time = evalin('base','exist(''Fs'',''var'')==1');
if use_time
    Fs = evalin('base','Fs');
    if ~isscalar(Fs) || ~isfinite(Fs) || Fs<=0
        warning('检测到无效的 Fs，改用样本索引。');
        use_time = false;
    end
end

% 批处理
for k = 1:numel(file_list)
    fname = fullfile(file_list(k).folder, file_list(k).name);
    fprintf('\n[%d/%d] 处理文件：%s\n', k, numel(file_list), file_list(k).name);

    try
        % 1) 自动检测“应跳过的表头行数”及“列名”
        [numHeaderLines, headerNames] = detectHeaderAndNames(fname);

        % 2) 读入数据体（至少6列）
        data = tryReadMatrix(fname, numHeaderLines);
        if size(data,2) < 6
            warning('列数 < 6，跳过此文件：%s', file_list(k).name);
            continue;
        end

        sigs = data(:,1:6);
        n = size(sigs,1);

        % 3) 横轴
        if use_time
            t = (0:n-1).' / Fs;
            xlab = sprintf('Time (s), Fs=%.6g', Fs);
            x = t;
        else
            x = (1:n).';
            xlab = 'Sample index';
        end

        % 4) 列名（无法解析时给默认名）
        defNames = {'Col1','Col2','Col3','Col4','Col5','Col6'};
        if isempty(headerNames)
            names = defNames;
        else
            % 只取前6个，且清理非法字符
            names = headerNames;
            for i = 1:min(6, numel(names))
                names{i} = cleanTitle(names{i});
            end
            if numel(names) < 6
                names(end+1:6) = defNames(numel(names)+1:6);
            end
        end

        % 5) 画图（六图同页）
        fig = figure('Visible','off'); % 如需显示，改为 'on'
        fig.Color = [1 1 1];
        fig.Position(3:4) = [1200 900];

        tl = tiledlayout(fig, 3, 2, 'TileSpacing','compact', 'Padding','compact');
        [~, shortName, ~] = fileparts(file_list(k).name);
        sgtitle(tl, sprintf('%s — 六列数据可视化', shortName), 'FontWeight','bold', 'Interpreter','none');

        for i = 1:6
            nexttile(tl, i);
            plot(x, sigs(:,i), 'LineWidth', 1.0);
            grid on;
            title(names{i}, 'Interpreter','none');
            xlabel(xlab);
            ylabel('Amplitude');
        end

        % 6) 保存
        png_name = fullfile(out_dir, sprintf('%s_sixcols.png', shortName));
        try
            exportgraphics(fig, png_name, 'Resolution', 200);
        catch
            saveas(fig, png_name); % 旧版兼容
        end
        % 可选保存 .fig
        % savefig(fig, fullfile(out_dir, sprintf('%s_sixcols.fig', shortName)));

        close(fig);
        fprintf('已输出：%s\n', png_name);
    catch ME
        warning('处理文件失败：%s\n原因：%s', file_list(k).name, ME.message);
        continue;
    end
end

fprintf('\n批处理完成。所有图像已保存在：%s\n', out_dir);


% ======================= 辅助函数区域 =======================
function [numHeaderLines, headerNames] = detectHeaderAndNames(fname)
% 自动检测应跳过的表头行数；尝试解析列名
    numHeaderLines = 0;
    headerNames = {};
    fid = fopen(fname,'r','n','UTF-8');
    if fid < 0
        error('无法打开文件（路径/编码问题？）');
    end
    cleaner = onCleanup(@() fclose(fid));

    % 读取前若干行，直到遇到“以数值开头”的行
    maxProbe = 20;
    lines = strings(0);
    while numHeaderLines < maxProbe
        pos = ftell(fid);
        ln = fgetl(fid);
        if ~ischar(ln) && ~isstring(ln)
            break; % EOF
        end
        lines(end+1) = string(ln); %#ok<AGROW>
        if startsWithNumeric(ln)
            % 回退到该行开头，用作数据第一行
            fseek(fid, pos, 'bof');
            break;
        else
            numHeaderLines = numHeaderLines + 1;
        end
    end

    % 尝试从“最近的一行非数字行”解析列名
    if ~isempty(lines)
        % 优先最后一行非数值行；也检查是否包含连字符分隔
        for idx = numel(lines):-1:1
            ln = strtrim(lines(idx));
            if ln == "", continue; end
            if ~startsWithNumeric(ln)
                headerNames = splitHeaderNames(ln);
                if ~isempty(headerNames), break; end
            end
        end
    end
end

function tf = startsWithNumeric(ln)
    if isempty(ln), tf = false; return; end
    ln = strtrim(string(ln));
    if ln == "", tf = false; return; end
    c = char(ln);
    tf = ~isempty(regexp(c,'^[\+\-]?(\d|\.|\d\.)','once')); % 以数字/小数点/符号+数字开头
end

function names = splitHeaderNames(ln)
% 尝试以常见分隔符切分列名：连字符、破折号、逗号、制表符、多个空格
    ln = regexprep(ln,'[—–-]+','---'); % 统一成 '---' 作为强分隔
    parts = regexp(ln, '\s*---\s*|\t|,\s*|\s{2,}', 'split');
    % 若仍只有1段，再退化用单空格分
    if numel(parts) <= 1
        parts = regexp(ln, '\s+', 'split');
    end
    % 清理空白、非字母数字字符收尾
    names = {};
    for i = 1:numel(parts)
        s = strtrim(parts{i});
        s = regexprep(s,'^[\s\p{Z}\p{C}]+|[\s\p{Z}\p{C}]+$',''); % 去除不可见字符
        if s ~= ""
            names{end+1} = s; %#ok<AGROW>
        end
    end
    if numel(names) < 2
        names = {}; % 解析失败时返回空
    end
end

function data = tryReadMatrix(fname, numHeaderLines)
% 优先 readmatrix，失败则 textscan
    try
        data = readmatrix(fname, 'NumHeaderLines', numHeaderLines);
        if isempty(data) || all(isnan(data(:)))
            data = readmatrix(fname, 'NumHeaderLines', numHeaderLines, ...
                              'Delimiter',' ', 'MultipleDelimsAsOne',true);
        end
    catch
        fid = fopen(fname,'r','n','UTF-8');
        if fid<0
            error('无法打开文件。');
        end
        cleaner = onCleanup(@() fclose(fid));
        % 跳过表头
        for i=1:numHeaderLines, fgetl(fid); end
        C = textscan(fid,'%f %f %f %f %f %f','CollectOutput',true);
        data = C{1};
    end
end

function s = cleanTitle(s)
% 供标题显示的简洁名称
    s = string(s);
    s = regexprep(s,'[_\-\s]+$','');
    s = char(s);
end
