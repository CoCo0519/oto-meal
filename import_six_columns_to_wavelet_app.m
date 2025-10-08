% import_six_columns_to_wavelet_app.m
% ---------------------------------------------------------------
% 一键：从含6列的TXT文件读取数据 -> 放入工作区 -> 逐列打开 Wavelet Signal Denoiser App
% One-click: Read 6-column TXT -> put variables into base workspace -> open Wavelet Signal Denoiser per column
% 适用：首行是列名（将被跳过），其余为以空格/制表符分隔的6列数值。
% Usage: Run the script, pick your TXT file (or set fname below).

% ========== 用户可选：手动指定文件路径（把下一行改成你的实际路径） ==========
fname = '"G:\Project-Swallow\denoised_hyx_data\喉咙-吞咽每5秒一次共31秒_approx.txt"';

% 若未设置 fname 或路径不存在，则弹窗选择文件
if ~exist('fname','var') || isempty(fname) || ~isfile(fname)
    [f,p] = uigetfile({'*.txt;*.dat','Text Files (*.txt, *.dat)'; '*.*','All Files (*.*)'}, ...
                      '请选择包含六列数据的文本文件 / Choose a 6-column TXT');
    if isequal(f,0)
        error('未选择文件 / No file selected.');
    end
    fname = fullfile(p,f);
end

fprintf('读取文件：%s\n', fname);

% ---------- 读入数据（跳过第一行列名） ----------
data = [];
try
    % 新版本MATLAB优先使用 readmatrix
    data = readmatrix(fname,'NumHeaderLines',1);            % 自动识别分隔符
    % 如果读到很多 NaN，尝试以空格作为多重分隔符重读
    if any(isnan(data(:)))
        data = readmatrix(fname,'NumHeaderLines',1, ...
                          'Delimiter',' ', 'MultipleDelimsAsOne',true);
    end
catch
    % 兼容老版本：使用 textscan
    fid = fopen(fname,'r','n','UTF-8');
    if fid<0
        error('无法打开文件（编码/路径问题？） / Cannot open file.');
    end
    fgetl(fid); % 跳过首行列名 / skip header line
    C = textscan(fid,'%f %f %f %f %f %f','CollectOutput',true);
    fclose(fid);
    data = C{1};
end

% 基本校验
if size(data,2) < 6
    error('检测到的列数 < 6，请检查文件格式 / Less than 6 columns detected.');
end

% 取前6列并命名
g  = data(:,1);   % 绿光 / Green
ir = data(:,2);   % 红外光 / Infrared
rd = data(:,3);   % 红光 / Red
X  = data(:,4);   % X轴 / X-axis
Y  = data(:,5);   % Y轴 / Y-axis
Z  = data(:,6);   % Z轴 / Z-axis

% 放入基础工作区（便于在APP中 Import from Workspace）
assignin('base','g',g);
assignin('base','ir',ir);
assignin('base','rd',rd);
assignin('base','X',X);
assignin('base','Y',Y);
assignin('base','Z',Z);

% 也提供一个带中文名的表（可选）
try
    T = table(g,ir,rd,X,Y,Z, 'VariableNames', {'绿光','红外光','红光','X轴','Y轴','Z轴'}); %#ok<NASGU>
    assignin('base','传感数据表',T); % 名称含中文，仅作为演示，可在变量区查看
catch
    % 老版本可能不支持中文变量名，忽略
end

fprintf('已将变量 g, ir, rd, X, Y, Z 放入工作区（base workspace）。\n');
fprintf('Now opening Wavelet Signal Denoiser app for each signal...\n');

% ---------- 打开 Wavelet Signal Denoiser 应用并载入每列 ----------
% 说明：某些版本仅支持把变量导入到App；多数版本也支持直接传入向量打开。
failed = false;
try, waveletSignalDenoiser(g);  catch, warning('无法直接为 g 打开App，请手动在App中 Import。'); failed = true; end
try, waveletSignalDenoiser(ir); catch, warning('无法直接为 ir 打开App，请手动在App中 Import。'); failed = true; end
try, waveletSignalDenoiser(rd); catch, warning('无法直接为 rd 打开App，请手动在App中 Import。'); failed = true; end
try, waveletSignalDenoiser(X);  catch, warning('无法直接为 X 打开App，请手动在App中 Import。'); failed = true; end
try, waveletSignalDenoiser(Y);  catch, warning('无法直接为 Y 打开App，请手动在App中 Import。'); failed = true; end
try, waveletSignalDenoiser(Z);  catch, warning('无法直接为 Z 打开App，请手动在App中 Import。'); failed = true; end

if failed
    fprintf(['提示：若当前MATLAB版本不支持用命令直接载入，请手动操作：\n' ...
             '1) 在命令行输入 waveletSignalDenoiser 打开应用；\n' ...
             '2) 点击左上角 Import > Workspace；\n' ...
             '3) 依次选择 g / ir / rd / X / Y / Z 并点击 Import。\n']);
else
    fprintf('已为6个信号各打开一个去噪App窗口。\n');
end

% （可选）如果你只想批量程序化去噪而不使用App，可参考：
% g_d  = wdenoise(g,  'DenoisingMethod','UniversalThreshold');
% ir_d = wdenoise(ir, 'DenoisingMethod','UniversalThreshold');
% rd_d = wdenoise(rd, 'DenoisingMethod','UniversalThreshold');
% X_d  = wdenoise(X,  'DenoisingMethod','UniversalThreshold');
% Y_d  = wdenoise(Y,  'DenoisingMethod','UniversalThreshold');
% Z_d  = wdenoise(Z,  'DenoisingMethod','UniversalThreshold');
% writematrix([g_d, ir_d, rd_d, X_d, Y_d, Z_d], 'denoised_6cols.txt');
