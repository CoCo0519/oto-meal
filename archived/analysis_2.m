%% 
% 放置在喉咙的IMU传感器和PPG传感器数据
% channel2是放在喉咙的，channel1是放在耳道的
filename = './hyx_data/耳道-吞咽6次间隔10秒.txt';

% 使用 readmatrix，并跳过第一行
data = readmatrix(filename, 'NumHeaderLines', 1);  % 从第二行开始读

% 显示结果
disp('读取的数据为：');
disp(data);

%% 基础切片（保持你的写法）
ax = data(1000:end, 4);
ay = data(1000:end, 5);
az = data(1000:end, 6);

ax = detrend(ax);
ay = detrend(ay);
az = detrend(az);

ppg_green = data(1000:end, 1);
ppg_IR    = data(1000:end, 2);
ppg_red   = data(1000:end, 3);

ppg_green = detrend(ppg_green);
ppg_IR    = detrend(ppg_IR);
ppg_red   = detrend(ppg_red);

%% 图1：与参考代码相同的subplot输出形式（每次都显式fs/butter/filtfilt/plot）
figure('Name','原始6通道（滤波后）');

fs = 100;  % 采样率（请根据你的设备实际值填写）
[b, a] = butter(4, [0.5 5]/(fs/2), 'bandpass');
ax_filt = filtfilt(b, a, ax);
subplot(2, 3, 1);
plot(ax_filt);
title('喉咙IMU-X轴'); grid on;

fs = 100;  % 采样率（请根据你的设备实际值填写）
[b, a] = butter(4, [0.5 5]/(fs/2), 'bandpass');
ay_filt = filtfilt(b, a, ay);
subplot(2, 3, 2);
plot(ay_filt);
title('喉咙IMU-Y轴'); grid on;

fs = 100;  % 采样率（请根据你的设备实际值填写）
[b, a] = butter(4, [0.5 5]/(fs/2), 'bandpass');
az_filt = filtfilt(b, a, az);
subplot(2, 3, 3);
plot(az_filt);
title('喉咙IMU-Z轴'); grid on;

fs = 100;  % 采样率（请根据你的设备实际值填写）
[b, a] = butter(4, [0.5 5]/(fs/2), 'bandpass');
ppg_green_filt = filtfilt(b, a, ppg_green);
subplot(2, 3, 4);
plot(ppg_green_filt);
title('喉咙PPG-green'); grid on;

fs = 100;  % 采样率（请根据你的设备实际值填写）
[b, a] = butter(4, [0.5 5]/(fs/2), 'bandpass');
ppg_IR_filt = filtfilt(b, a, ppg_IR);
subplot(2, 3, 5);
plot(ppg_IR_filt);
title('喉咙PPG-IR'); grid on;

fs = 100;  % 采样率（请根据你的设备实际值填写）
[b, a] = butter(4, [0.5 5]/(fs/2), 'bandpass');
ppg_red_filt = filtfilt(b, a, ppg_red);
subplot(2, 3, 6);
plot(ppg_red_filt);
title('喉咙PPG-red'); grid on;

%% —— 新增：吞咽段提取 + 对齐叠加 + 平均/拼接 —— 
% 已知吞咽时刻（秒）
swallow_sec = [10 20 30 40 50 60];

% 如果录制不是从 0s 开始，请设置录制起始时间（秒）
% 例如录制从 2.3s 开始，改为 recording_start_sec = 2.3;
recording_start_sec = 0;

% 窗长设置（相对吞咽时刻的前后秒数）
pre_sec  = 1.5;
post_sec = 2.5;

% "剪去前1000行"的偏移换算到秒；用于把时间转成当前向量索引
trim_offset_samples = 1000 - 1;         % 原始第1000行 -> 新向量第1个样本
trim_offset_sec     = trim_offset_samples / fs;

% 吞咽事件在当前向量中的索引（从1开始）
event_idx = round( (swallow_sec - recording_start_sec - trim_offset_sec) * fs ) + 1;

% 构造统一的时间窗索引
preN  = round(pre_sec  * fs);
postN = round(post_sec * fs);
winN  = preN + postN + 1;               % 每段窗样本数
t_win = (-preN:postN) / fs;             % 对齐时间轴（吞咽时刻为0）

% 从每个通道提取窗段（自动越界处理 + Z-score 归一化）
seg_ax = local_extract(ax_filt, event_idx, preN, postN);
seg_ay = local_extract(ay_filt, event_idx, preN, postN);
seg_az = local_extract(az_filt, event_idx, preN, postN);
seg_pg = local_extract(ppg_green_filt, event_idx, preN, postN);
seg_ir = local_extract(ppg_IR_filt,    event_idx, preN, postN);
seg_pr = local_extract(ppg_red_filt,   event_idx, preN, postN);

%% 图2：对齐叠加 + 平均（仍然用 subplot(2,3,i)）
figure('Name','吞咽段对齐叠加与平均(6通道)');

subplot(2,3,1); local_plot_overlaid_subplot(t_win, seg_ax, '吞咽段-IMU X');
subplot(2,3,2); local_plot_overlaid_subplot(t_win, seg_ay, '吞咽段-IMU Y');
subplot(2,3,3); local_plot_overlaid_subplot(t_win, seg_az, '吞咽段-IMU Z');
subplot(2,3,4); local_plot_overlaid_subplot(t_win, seg_pg, '吞咽段-PPG green');
subplot(2,3,5); local_plot_overlaid_subplot(t_win, seg_ir, '吞咽段-PPG IR');
subplot(2,3,6); local_plot_overlaid_subplot(t_win, seg_pr, '吞咽段-PPG red');

%% 图3："拼接"序列（把6个窗段顺次相连）——用同样的 subplot 形式
concat_ax = local_concat(seg_ax);
concat_ay = local_concat(seg_ay);
concat_az = local_concat(seg_az);
concat_pg = local_concat(seg_pg);
concat_ir = local_concat(seg_ir);
concat_pr = local_concat(seg_pr);

figure('Name','吞咽段顺序拼接（6通道）');
subplot(2,3,1); plot(concat_ax); title('拼接-IMU X'); grid on; xlim([1 numel(concat_ax)]);
subplot(2,3,2); plot(concat_ay); title('拼接-IMU Y'); grid on; xlim([1 numel(concat_ay)]);
subplot(2,3,3); plot(concat_az); title('拼接-IMU Z'); grid on; xlim([1 numel(concat_az)]);
subplot(2,3,4); plot(concat_pg); title('拼接-PPG green'); grid on; xlim([1 numel(concat_pg)]);
subplot(2,3,5); plot(concat_ir); title('拼接-PPG IR'); grid on; xlim([1 numel(concat_ir)]);
subplot(2,3,6); plot(concat_pr); title('拼接-PPG red'); grid on; xlim([1 numel(concat_pr)]);

% 如需导出文本/Mat，可直接：
% writematrix(concat_ax, 'concat_ax.txt');
% save('swallow_segments.mat','t_win','seg_ax','seg_ay','seg_az','seg_pg','seg_ir','seg_pr','concat_ax','concat_ay','concat_az','concat_pg','concat_ir','concat_pr');

%% ====== 本脚本内的小函数（置于文件末尾即可） ======
function seg = local_extract(sig, event_idx, preN, postN)
    % sig: 列向量
    % 返回 seg: 大小 [winN × 次数]，每列是一次吞咽的窗段（Z-score 归一化）
    L = numel(sig);
    winN = preN + postN + 1;
    K = numel(event_idx);
    seg = nan(winN, K);
    for k = 1:K
        idx0 = event_idx(k);
        s = idx0 - preN;
        e = idx0 + postN;
        if s < 1 || e > L
            % 边界不足则跳过（保留 NaN）
            continue;
        end
        w = sig(s:e);
        % 为便于跨次比较和平均，做 Z-score（如需保留原幅值，可改成仅减均值/不归一）
        mu = mean(w);
        sd = std(w);
        w = (w - mu) / (sd + eps);
        seg(:, k) = w(:);
    end
    % 去掉全 NaN 的列（若越界导致）
    if ~isempty(seg)
        allNaN = all(isnan(seg), 1);
        seg(:, allNaN) = [];
    end
end

function local_plot_overlaid_subplot(t_win, seg_mat, ttl)
    % 在当前subplot内绘制6次叠加 + 平均（不使用tiledlayout）
    hold on;
    % 每一次
    for i = 1:size(seg_mat,2)
        plot(t_win, seg_mat(:,i), 'LineWidth', 0.8);
    end
    % 平均
    mu = nanmean(seg_mat, 2);
    plot(t_win, mu, 'k', 'LineWidth', 2);  % 平均曲线加粗
    xline(0, '--');
    grid on;
    xlabel('时间 (s)'); ylabel('归一化幅值');
    title(ttl);
    hold off;
end

function catv = local_concat(seg_mat)
    % 把列向量窗段顺序拼接成一条长序列
    if isempty(seg_mat)
        catv = [];
        return;
    end
    tmp = seg_mat;
    tmp(isnan(tmp)) = 0;   % 用 0 代替 NaN（也可改为插值）
    catv = tmp(:);
end
