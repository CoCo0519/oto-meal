function view_six_signals_from_mat_multi(mat_files)
% 六通道查看器(深色主题)——强制英文通道名顺序：g, ir, rd, imu_x, imu_y, imu_z
% Six-channel viewer (dark theme) — forced channel order: g, ir, rd, imu_x, imu_y, imu_z
%
% 用法 / Usage:
%   view_six_signals_from_mat_multi
%   view_six_signals_from_mat_multi('C:\a.mat')
%   view_six_signals_from_mat_multi({'C:\a.mat','D:\b.mat'})

%% ===== 主题 & 样式 / Theme & style =====
THEME      = 'dark';      % 'dark' or 'light'
FONT_SIZE  = 12;
LINE_WIDTH = 1.4;

FORCED_NAMES = {'g','ir','rd','imu_x','imu_y','imu_z'};

%% ===== 选择文件 / File pick =====
if nargin<1 || isempty(mat_files)
    [f,p] = uigetfile({'*.mat','MAT-files (*.mat)'}, ...
        'Select one or more MAT files','MultiSelect','on');
    if isequal(f,0), error('No file selected.'); end
    if iscell(f), mat_files = fullfile(p,f); else, mat_files = {fullfile(p,f)}; end
elseif ischar(mat_files) || (isstring(mat_files) && isscalar(mat_files))
    mat_files = {char(mat_files)};
elseif ~iscell(mat_files)
    error('Argument must be a string path or a cell array of paths.');
end

%% ===== 每个文件单独开窗 / One window per file =====
for idx = 1:numel(mat_files)
    mf = mat_files{idx};
    if ~isfile(mf), warning('File not found: %s', mf); continue; end

    % ---- 加载 / Load ----
    M = load(mf);
    root = M;
    fn = fieldnames(M);
    if numel(fn)==1 && isstruct(M.(fn{1}))
        root = M.(fn{1});
    end

    % ---- 从 root 中尽可能按名称抓取 6 通道 / Try to fetch by names ----
    [sigs, names] = prefer_forced_names(root, FORCED_NAMES);

    % 退化：若没抓到，就在任意 N×6 数值矩阵中取用，并强制名字
    % Fallback: if not found, search any N×6 numeric matrix and force names
    if isempty(sigs)
        [sigs, ~] = find_numeric_6cols(root);
        if isempty(sigs)
            error('Could not detect 6 channels in: %s', mf);
        end
        names = FORCED_NAMES;
    end

    N = size(sigs,1);

    % ---- 时间轴 / Time axis ----
    [t, xlab] = build_timebase(root, N);
    if isempty(t) && evalin('base','exist(''t'',''var'')==1')
        tb = evalin('base','t'); if isnumeric(tb)&&isvector(tb)&&numel(tb)==N, t=tb(:); xlab='Time (s)'; end
    end
    if isempty(t) && evalin('base','exist(''Fs'',''var'')==1')
        Fs = evalin('base','Fs'); if isnumeric(Fs)&&isscalar(Fs)&&Fs>0, t=(0:N-1)'/Fs; xlab=sprintf('Time (s), Fs=%.6g',Fs); end
    end
    if isempty(t), x = (1:N)'; xlab='Sample index'; else, x=t; end

    % ---- 主题颜色 / Theme colors ----
    switch lower(THEME)
        case 'dark'
            figColor  = [0.10 0.11 0.12];
            axColor   = [0.13 0.14 0.16];
            gridColor = [0.35 0.35 0.40];
            textColor = [0.95 0.95 0.96];
        otherwise
            figColor  = [1 1 1];
            axColor   = [1 1 1];
            gridColor = [0.70 0.70 0.70];
            textColor = [0 0 0];
    end

    % ---- 绘图 / Plot ----
    [~, shortName, ~] = fileparts(mf);
    fig = figure('Color',figColor,'Position',[120 80 1250 900], ...
        'Name',['Six-channel viewer: ' shortName], 'NumberTitle','off');
    tl = tiledlayout(fig,3,2,'TileSpacing','compact','Padding','compact');
    sgtitle(tl, sprintf('%s — Six-channel view', shortName), ...
        'Interpreter','none','FontWeight','bold','Color',textColor);

    axList = gobjects(1,6);
    for i = 1:6
        ax = nexttile(tl,i);
        ax.Color     = axColor;
        ax.GridColor = gridColor;
        ax.XColor    = textColor;
        ax.YColor    = textColor;
        ax.LineWidth = 1;
        ax.FontSize  = FONT_SIZE;
        grid(ax,'on');
        plot(ax, x, sigs(:,i), 'LineWidth', LINE_WIDTH);
        title(ax, names{i}, 'Interpreter','none','Color',textColor,'FontWeight','bold');
        xlabel(ax, xlab, 'Color',textColor);
        ylabel(ax, 'Amplitude', 'Color',textColor);
        axList(i) = ax;
    end
    linkaxes(axList,'x');

    % ---- 保存 / Save ----
    out_dir = fullfile(fileparts(mf),'fig_out');
    if ~exist(out_dir,'dir'), mkdir(out_dir); end
    png_name = fullfile(out_dir, sprintf('%s_view6_%s.png', shortName, THEME));
    try, exportgraphics(fig, png_name, 'Resolution', 220); catch, saveas(fig, png_name); end
    fprintf('Saved: %s\n', png_name);
end
end

% ================= Helpers =================
function [t, xlab] = build_timebase(root, N)
t=[]; xlab='';
for key = {'t','time','Time'}
    if isfield(root,key{1})
        tv = root.(key{1});
        if isnumeric(tv)&&isvector(tv)&&numel(tv)==N, t=tv(:); xlab='Time (s)'; return; end
    end
end
for key = {'Fs','fs','sample_rate','sampling_rate'}
    if isfield(root,key{1})
        Fs = root.(key{1});
        if isnumeric(Fs)&&isscalar(Fs)&&isfinite(Fs)&&Fs>0
            t=(0:N-1)'/Fs; xlab=sprintf('Time (s), Fs=%.6g',Fs); return;
        end
    end
end
end

function [sigs, names] = find_numeric_6cols(root)
sigs=[]; names={};
if isnumeric(root) && size(root,2)==6 && size(root,1)>10
    sigs = root; names = repmat({''},1,6); return;
end
if isstruct(root)
    fn = fieldnames(root);
    for i=1:numel(fn)
        a = root.(fn{i});
        if isnumeric(a) && ismatrix(a) && size(a,2)==6 && size(a,1)>10
            sigs = a; names = repmat({''},1,6); return;
        end
    end
end
end

function [sigs, names] = prefer_forced_names(root, forced)
% 尽量按 forced 名称抓取；支持顶层 struct 或内部一层 struct/table
sigs=[]; names=forced;
ctxs = {root};
if isstruct(root) && numel(fieldnames(root))==1
    fn = fieldnames(root);
    if isstruct(root.(fn{1})), ctxs{end+1} = root.(fn{1}); end
end
for c = 1:numel(ctxs)
    ctx = ctxs{c};
    ok = true; bag = zeros(common_length(ctx,forced), 6);
    for j = 1:6
        v = fetch_one(ctx, forced{j});
        if isempty(v), ok = false; break; end
        L = size(bag,1);
        bag(:,j) = v(1:L);
    end
    if ok, sigs = bag; return; end
end
end

function L = common_length(ctx, forced)
L = inf;
for j=1:6
    v = fetch_one(ctx, forced{j});
    if isempty(v), L=0; return; end
    L = min(L, numel(v));
end
if isinf(L), L=0; end
end

function v = fetch_one(ctx, key)
v = [];
if ~isstruct(ctx), return; end
if isfield(ctx, key)
    raw = ctx.(key);
    v = to_col(raw);
    return;
end
% 一层嵌套里找
fn = fieldnames(ctx);
for i=1:numel(fn)
    node = ctx.(fn{i});
    if isstruct(node) && isfield(node, key)
        v = to_col(node.(key)); if ~isempty(v), return; end
    elseif istable(node) && any(strcmp(node.Properties.VariableNames, key))
        v = to_col(node.(key)); if ~isempty(v), return; end
    end
end
end

function v = to_col(x)
v = [];
if isnumeric(x) && isvector(x), v=x(:); end
if istable(x) && width(x)==1
    a = x{:,1}; if isnumeric(a)&&isvector(a), v=a(:); end
end
if iscell(x) && numel(x)==1 && isnumeric(x{1}) && isvector(x{1})
    v = x{1}(:);
end
end
