@echo off
REM ==== 10 折交叉验证（仅耳道）====
REM 每一折都会训练较久，耗时更长；得到聚合混淆矩阵（样本数很大）与平均准确率

python run_behavior_classification.py ^
  --data-dir .\denoised_hyx_data ^
  --event-config .\events_config.json ^
  --epochs 80 ^
  --batch-size 192 ^
  --lr 3e-4 ^
  --weight-decay 1e-2 ^
  --emb-dim 256 ^
  --nlayers 6 ^
  --nhead 8 ^
  --dropout 0.2 ^
  --cv-folds 10 ^
  --aug

pause
