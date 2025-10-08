function inspect_mat(matPath, opts)
% 查看 .mat：列出变量，并分图显示 approx / data / data_raw（每份六分图，多页翻）。
% If approx is absent, compute it on the fly from data using meta (wavelet='sym8', level=7 by default).
%
% 用法：
%   inspect_mat;
%   inspect_mat('G:\Project-Swallow\denoised_hyx_data\xxx_denoise.mat');
%   inspect_mat('...\file.mat', struct('Hz',100,'AssignToBase',false));
%
% 选项：
%   opts.Hz           —— 若无 t，则用 Hz 生成时间（默认 100）
%   opts.AssignToBase —— 是否把变量导入 base 工作区（默认 false）

arguments
    matPath (1,:) char = ""
    opts.Hz (1,1) double {mustBePositive} = 100
    opts.AssignToBase (1,1) logical = false
end

% 选择文件
if isempty(matPath)
    [f,p] = uigetfile({'*.mat','MAT-files (*.mat)'}, 'Select a MAT file');
    if isequal(f,0), disp('Canceled.'); return; end
    matPath = fullfile(p,f);
end
if ~isfile(matPath), error('File not found: %s', matPath); end

fprintf('>> Inspecting: %s\n', matPath);

% 只列变量
info = whos('-file', matPath);
if ~isempty(info)
    fprintf('\n== Variables ==\n%-24s %-12s %-10s\n','Name','Size','Class');
    for k = 1:numel(info)
        s = sprintf('%dx%d', info(k).size(1), ...
            max(1, prod(info(k).size)/info(k).size(1)));
        fprintf('%-24s %-12s %-10s\n', info(k).name, s, info(k).class);
    end
else
    warning('No variables in file.');
end

% 载入
S = load(matPath);
if opts.AssignToBase
    fns = fieldnames(S);
    for i = 1:numel(fns), assignin('base', fns{i}, S.(fns{i})); end
    fprintf('\n(Assigned to base workspace.)\n');
end

% 取 t/Fs
t = [];
Fs = [];
if isfield(S,'t'), t = S.t; end
if isfield(S,'meta') && isstruct(S.meta)
    if isempty(t) && isfield(S.meta,'t'), t = S.meta.t; end
    if isfield(S.meta,'Fs'), Fs = S.meta.Fs; end
end
if isempty(Fs), Fs = opts.Hz; end

% 列名
names = [];
if isfield(S,'names'), names = S.names; end

% 近似优先显示：若没有 approx，就从 data 现场计算
if isfield(S,'approx') && isnumeric(S.approx) && ndims(S.approx)==2
    plot_six(S.approx, t, Fs, names, 'approx (A7)');
elseif isfield(S,'data') && isnumeric(S.data) && ndims(S.data)==2
    % 在线计算 A7：使用 meta 参数（若不存在，则默认 'sym8', level=7）
    [wv, lv] = get_params(S);
    A = compute_approx_from_data(S.data, wv, lv);
    plot_six(A, t, Fs, names, sprintf('approx (A%d) [computed]', lv));
else
    fprintf('\n(No approx/data to compute approximation from.)\n');
end

% 再画去噪数据
if isfield(S,'data') && isnumeric(S.data) && ndims(S.data)==2
    plot_six(S.data, t, Fs, names, 'data (denoised)');
end

% 再画原始数据
if isfield(S,'data_raw') && isnumeric(S.data_raw) && ndims(S.data_raw)==2
    plot_six(S.data_raw, t, Fs, names, 'data_raw (raw)');
end
end

% ===== helpers =====
function [wname, level] = get_params(S)
wname = 'sym8'; level = 7;
if isfield(S,'meta') && isstruct(S.meta) && isfield(S.meta,'params')
    P = S.meta.params;
    if isfield(P,'wavelet'), wname = P.wavelet; end
    if isfield(P,'level'),   level = P.level;   end
end
end

function A = compute_approx_from_data(M, wname, level)
[n,m] = size(M);
A = zeros(n,m);
for i = 1:m
    [cvec,lvec] = wavedec(M(:,i), level, wname);
    A(:,i) = wrcoef('a', cvec, lvec, wname, level);
end
end

function plot_six(M, t, Fs, names, ttl)
if isempty(M) || ~isnumeric(M), return; end
[n,m] = size(M);
tt = [];
if ~isempty(t) && isnumeric(t) && isvector(t) && numel(t)==n
    tt = t(:);
else
    tt = (0:n-1)'/Fs;
end

colsPerPage = 6;
numPages = ceil(m/colsPerPage);
for pg = 1:numPages
    idx = (pg-1)*colsPerPage + (1:colsPerPage);
    idx = idx(idx<=m);

    rows = (numel(idx) <= 3) * 1 + (numel(idx) > 3) * 2;
    cols = min(3, numel(idx));
    if rows == 0, continue; end

    f = figure('Name', sprintf('%s  [%d/%d]', ttl, pg, numPages), ...
               'Color','w','NumberTitle','off');
    tlo = tiledlayout(rows, 3, 'TileSpacing','compact','Padding','compact');

    for k = 1:numel(idx)
        ax = nexttile(tlo, k);
        plot(ax, tt, M(:,idx(k)), 'LineWidth', 0.9);
        if ~isempty(names) && numel(names)>=idx(k)
            title(ax, string(names{idx(k)}), 'Interpreter','none');
        else
            title(ax, sprintf('col%d', idx(k)));
        end
        grid(ax,'on');
        if k <= 3 && rows==2, ax.XTickLabel = []; end
    end
    title(tlo, sprintf('%s  [%d/%d]', ttl, pg, numPages), 'Interpreter','none');
    xlabel(tlo,'Time (s)'); ylabel(tlo,'Amplitude');
end
end
