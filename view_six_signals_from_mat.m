% view_six_signals_from_mat.m
% ------------------------------------------------------------
% 作用：打开一个 .mat 文件，自动识别与“6通道（如 绿光/红外/红光/X/Y/Z）”
%      类似的数据布局，并在同一页面（3x2）绘制六个子图。
% Purpose: Open a .mat file, auto-detect a 6-channel dataset (e.g., G/IR/R/X/Y/Z)
%          and plot six subplots (3x2) on a single page.
%
% 用法（Usage）:
%   直接运行本脚本；若未设置 mat_file，将弹出文件选择对话框。
%   Run this script; if 'mat_file' is not set, a file picker will appear.
%
% 可选（Optional）:
%   在工作区设置 Fs=采样率（Hz）或 t=时间向量（长度与信号一致）。
%   Set 'Fs' (Hz) in base workspace or 't' time vector (same length as signals).
%
% 兼容性（Compatibility）: R2019b+ (使用 tiledlayout/exportgraphics；旧版可改为 subplot/saveas)
%
% 作者注（Note）:
%   - 尽量自适应以下情况：
%     1) 顶层就有 g/ir/rd/X/Y/Z 向量；
%     2) 顶层只有一个结构体 S，信号在 S 的字段里（如 S.g, S.ir...）；
%     3) 顶层存在一个 N×6 的数值矩阵（如 data、M、signals 等）。
%   - 变量名支持中英文别名（如 “绿光/红外光/红光/X轴/Y轴/Z轴”）。
%
clear; clc;

% ============ 用户可选：手动指定 .mat 文件路径（把下一行改成你的路径） ============
% mat_file = 'C:\your\path\喉咙- 咳嗽共6次间隔10秒_denoise.mat';

% 若未设置路径，则弹窗选择
if ~exist('mat_file','var') || isempty(mat_file) || ~isfile(mat_file)
    [f,p] = uigetfile({'*.mat','MAT-files (*.mat)'}, '请选择含6通道数据的 .mat 文件 / Choose a 6-channel MAT file');
    if isequal(f,0), error('未选择文件 / No file selected.'); end
    mat_file = fullfile(p,f);
end

fprintf('加载文件 / Loading: %s\n', mat_file);
vars = whos('-file', mat_file);

% 先加载到结构 M（不污染工作区） / Load into struct M (avoid polluting base workspace)
M = load(mat_file);

% 可能存在“只有一个结构体”的情况：取该结构体作为根 / If single struct at top-level, use it as root
root = M;
fieldNames = fieldnames(M);
if numel(fieldNames)==1 && isstruct(M.(fieldNames{1}))
    root = M.(fieldNames{1});
end

% --------- 候选名列表（中英文与常用别名）/ Candidate name lists ---------
G_names  = lower_list({'g','green','绿光','pgreen','sig_g','ch1','col1','col_1','col01'});
IR_names = lower_list({'ir','infrared','红外','红外光','sig_ir','ch2','col2','col_2','col02'});
RD_names = lower_list({'rd','red','红光','sig_rd','ch3','col3','col_3','col03'});
X_names  = lower_list({'x','ax','accx','imu_x','imuX','gx','wx','x轴','ch4','col4','col_4','col04'});
Y_names  = lower_list({'y','ay','accy','imu_y','imuY','gy','wy','y轴','ch5','col5','col_5','col05'});
Z_names  = lower_list({'z','az','accz','imu_z','imuZ','gz','wz','z轴','ch6','col6','col_6','col06'});

% --------- 从根结构中提取 6 通道 / Extract six channels ---------
[sigs, names, src] = extract_six(root, G_names, IR_names, RD_names, X_names, Y_names, Z_names);

% 如果未找到，尝试在顶层扫描“任意 N×6 数值矩阵” / Otherwise search for any N×6 numeric matrix
if isempty(sigs)
    [sigs, names, src] = find_numeric_6cols(root);
end

if isempty(sigs)
    error('未能自动识别 6 通道数据。请检查 .mat 结构或手动指定变量。 / Failed to auto-detect 6 channels.');
end

N = size(sigs,1);

% --------- 时间轴：优先 t，其次 Fs，否则样本索引 / Time base: t > Fs > samples ---------
[t, xlab] = build_timebase(root, N);
if isempty(t)
    % 也检查 base 工作区的 Fs/t / Also check base workspace Fs/t
    if evalin('base','exist(''t'',''var'')==1')
        t = evalin('base','t');
        if numel(t)~=N, warning('base.t 长度与信号不一致，改用样本索引。'); t = []; end
    end
end
if isempty(t)
    if evalin('base','exist(''Fs'',''var'')==1')
        Fs = evalin('base','Fs');
        if isscalar(Fs) && isfinite(Fs) && Fs>0
            t = (0:N-1).'/Fs; xlab = sprintf('Time (s), Fs=%.6g', Fs);
        end
    end
end
if isempty(t)
    x = (1:N).'; xlab = 'Sample index';
else
    x = t;
end

% --------- 绘图 / Plot ---------
fig = figure('Color',[1 1 1], 'Position',[100 100 1200 900]);
tl = tiledlayout(fig, 3, 2, 'TileSpacing','compact', 'Padding','compact');
[~, shortName, ~] = fileparts(mat_file);
sgtitle(tl, sprintf('%s — 六通道查看 / Six-channel view', shortName), 'FontWeight','bold', 'Interpreter','none');

for i = 1:6
    nexttile(tl,i);
    plot(x, sigs(:,i), 'LineWidth', 1.0);
    grid on;
    title(names{i}, 'Interpreter','none');
    xlabel(xlab);
    ylabel('Amplitude');
end
linkaxes(findall(fig,'Type','axes'), 'x');

% 输出目录与保存 / Output dir & save
out_dir = fullfile(fileparts(mat_file), 'fig_out');
if ~exist(out_dir,'dir'), mkdir(out_dir); end
png_name = fullfile(out_dir, sprintf('%s_view6.png', shortName));
try
    exportgraphics(fig, png_name, 'Resolution', 200);
catch
    saveas(fig, png_name);
end
fprintf('已保存图像 / Saved figure: %s\n', png_name);

% ======================== 辅助函数 / Helpers ========================
function L = lower_list(C)
    L = lower(string(C));
end

function [sigs, names, src] = extract_six(root, G_names, IR_names, RD_names, X_names, Y_names, Z_names)
% 在结构 root 中按候选名抓取 6 路信号；支持顶层或“仅一层嵌套结构”。
% Extract G/IR/R/X/Y/Z by candidate names; support top-level or one-level nested struct.
    sigs = [];
    names = {};
    src = 'named_fields';

    ctxs = {root};
    if isstruct(root) && numel(fieldnames(root))==1
        fn = fieldnames(root);
        if isstruct(root.(fn{1})), ctxs{end+1} = root.(fn{1}); end
    end

    % 尝试从多个上下文依次匹配 / try contexts in order
    for c = 1:numel(ctxs)
        ctx = ctxs{c};
        [g, gname]   = pick_one(ctx, G_names);
        [ir, irname] = pick_one(ctx, IR_names);
        [rd, rdname] = pick_one(ctx, RD_names);
        [X, Xname]   = pick_one(ctx, X_names);
        [Y, Yname]   = pick_one(ctx, Y_names);
        [Z, Zname]   = pick_one(ctx, Z_names);

        if ~isempty(g) && ~isempty(ir) && ~isempty(rd) && ~isempty(X) && ~isempty(Y) && ~isempty(Z)
            % 对齐长度 / align lengths
            N = min([numel(g),numel(ir),numel(rd),numel(X),numel(Y),numel(Z)]);
            sigs = [g(:), ir(:), rd(:), X(:), Y(:), Z(:)];
            sigs = sigs(1:N, :);
            names = {gname, irname, rdname, Xname, Yname, Zname};
            return;
        end
    end
end

function [y, pickedName] = pick_one(ctx, nameList)
% 在 ctx 中按候选名匹配向量；大小 >= 10 的数值列视为有效；大小 >2D 的忽略。
% Match a numeric vector by candidate names; length>=10 qualifies; >2D ignored.
    y = [];
    pickedName = '';
    fn = fieldnames(ctx);
    lowerMap = containers.Map(lower(fn), fn); % lower->actual

    % 先直接候选名匹配 / direct candidate match
    for i = 1:numel(nameList)
        key = char(nameList(i));
        if isKey(lowerMap, key)
            raw = ctx.(lowerMap(key));
            yvec = try_vectorize(raw);
            if ~isempty(yvec) && numel(yvec)>=10
                y = yvec;
                pickedName = lowerMap(key);
                return;
            end
        end
    end

    % 退化：若存在 table/struct 中的同名字段 / fallback for table/struct field
    for i = 1:numel(nameList)
        key = char(nameList(i));
        if isKey(lowerMap, key)
            raw = ctx.(lowerMap(key));
            if istable(raw) && any(strcmpi(raw.Properties.VariableNames, key))
                yvec = raw.(raw.Properties.VariableNames{strcmpi(raw.Properties.VariableNames, key)});
                yvec = try_vectorize(yvec);
                if ~isempty(yvec) && numel(yvec)>=10
                    y = yvec; pickedName = [lowerMap(key) '.' key]; return;
                end
            elseif isstruct(raw) && isfield(raw, key)
                yvec = try_vectorize(raw.(key));
                if ~isempty(yvec) && numel(yvec)>=10
                    y = yvec; pickedName = [lowerMap(key) '.' key]; return;
                end
            end
        end
    end
end

function v = try_vectorize(x)
% 将 x 尽量转换为列向量（数值/单列table/单元等）
% Try to coerce x to a numeric column vector.
    v = [];
    if isnumeric(x) && isvector(x)
        v = x(:); return;
    end
    if istable(x) && width(x)==1
        a = x{:,1};
        if isnumeric(a) && isvector(a), v = a(:); return; end
    end
    if iscell(x) && numel(x)==1 && isnumeric(x{1}) && isvector(x{1})
        v = x{1}(:); return;
    end
end

function [sigs, names, src] = find_numeric_6cols(root)
% 在 root 的字段中寻找 N×6 数值矩阵（N>>6），取首个满足者。
% Search for an N×6 numeric matrix (N>>6).
    sigs = [];
    names = {};
    src = 'matrix6';

    if isnumeric(root) && size(root,2)==6 && size(root,1) > 50
        sigs = root; names = default_names(); return;
    end

    if isstruct(root)
        fn = fieldnames(root);
        for i = 1:numel(fn)
            a = root.(fn{i});
            if isnumeric(a) && ndims(a)==2 && size(a,2)==6 && size(a,1) > 50
                sigs = a;
                names = default_names();
                return;
            end
        end
    end
end

function NAMES = default_names()
    NAMES = {'Col1','Col2','Col3','Col4','Col5','Col6'};
end

function [t, xlab] = build_timebase(root, N)
% 从 root 中提取 t 向量或 Fs 标量；都不存在则返回空。
% Extract time vector t or scalar Fs; otherwise return empty.
    t = [];
    xlab = '';

    % 1) 直接找 t（大小匹配）/ prefer 't'
    cand_t = {'t','time','Time','时间'};
    for i = 1:numel(cand_t)
        if isfield(root, cand_t{i})
            tv = root.(cand_t{i});
            if isnumeric(tv) && isvector(tv) && numel(tv)==N
                t = tv(:);
                xlab = 'Time (s)';
                return;
            end
        end
    end

    % 2) Fs -> 由 Fs 构造 t / Fs to build t
    cand_fs = {'Fs','fs','sample_rate','sampling_rate','Fs_Hz'};
    for i = 1:numel(cand_fs)
        if isfield(root, cand_fs{i})
            Fs = root.(cand_fs{i});
            if isnumeric(Fs) && isscalar(Fs) && isfinite(Fs) && Fs>0
                t = (0:N-1).'/Fs;
                xlab = sprintf('Time (s), Fs=%.6g', Fs);
                return;
            end
        end
    end
end
