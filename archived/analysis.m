%% 
% 放置在喉咙的IMU传感器和PPG传感器数据
% channel2是放在喉咙的，channel1是放在耳道的
%filename = './lhr/PPG_2_20250721_161608.txt';

filename = './hyx_data/喉咙-说话共6次持续5秒间隔10秒.txt';
%filename = './hyx_data/喉咙-吞咽6次间隔10秒.txt';

% 使用 readmatrix，并跳过第一行
data = readmatrix(filename, 'NumHeaderLines', 1);  % 从第二行开始读

% 显示结果
disp('读取的数据为：');
disp(data);

% plot 三轴
ax = data(1000:end, 4);
ay = data(1000:end, 5);
az = data(1000:end, 6);

ax = detrend(ax);
ay = detrend(ay);
az = detrend(az);

ppg_green = data(1000:end, 1);
ppg_IR = data(1000:end, 2);
ppg_red = data(1000:end, 3);

ppg_green = detrend(ppg_green);
ppg_IR = detrend(ppg_IR);
ppg_red = detrend(ppg_red);


fs = 100;  % 采样率（请根据你的设备实际值填写）
[b, a] = butter(4, [0.5 5]/(fs/2), 'bandpass');
ax_filt = filtfilt(b, a, ax);
subplot(2, 3, 1);
plot(ax_filt);
title('喉咙IMU-X轴');

fs = 100;  % 采样率（请根据你的设备实际值填写）
[b, a] = butter(4, [0.5 5]/(fs/2), 'bandpass');
ay_filt = filtfilt(b, a, ay);
subplot(2, 3, 2);
plot(ay_filt);
title('喉咙IMU-Y轴');

fs = 100;  % 采样率（请根据你的设备实际值填写）
[b, a] = butter(4, [0.5 5]/(fs/2), 'bandpass');
az_filt = filtfilt(b, a, az);
subplot(2, 3, 3);
plot(az_filt);
title('喉咙IMU-Z轴');


fs = 100;  % 采样率（请根据你的设备实际值填写）
[b, a] = butter(4, [0.5 5]/(fs/2), 'bandpass');
ppg_green_filt = filtfilt(b, a, ppg_green);
subplot(2, 3, 4);
plot(ppg_green_filt);
title('喉咙PPG-green');

fs = 100;  % 采样率（请根据你的设备实际值填写）
[b, a] = butter(4, [0.5 5]/(fs/2), 'bandpass');
ppg_IR_filt = filtfilt(b, a, ppg_IR);
subplot(2, 3, 5);
plot(ppg_IR_filt);
title('喉咙PPG-IR');

fs = 100;  % 采样率（请根据你的设备实际值填写）
[b, a] = butter(4, [0.5 5]/(fs/2), 'bandpass');
ppg_red_filt = filtfilt(b, a, ppg_red);
subplot(2, 3, 6);
plot(ppg_red_filt);
title('喉咙PPG-red');





