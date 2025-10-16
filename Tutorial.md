# Tutorial For In-Ear Wearable Eating Monitoring System

Github: [oto-meal](https://github.com/tsawke/oto-meal.git)

Latest package: [URL for Synology NAS](https://gofile.me/7xDmN/6LaSVzQNf)

## batch_export_wdenoise_denoise_only.m

Implement wavelet approximation and generate both .txt and .mat.

Run like:

```matlab
batch_export_wdenoise_denoise_only('G:\Project-Swallow\hyx_data', ...
    'G:\Project-Swallow\denoised_hyx_data', ...
    'HeaderLines',1,'Recursive',true, ...
    'Level',4,'Wavelet','sym8', ...
    'Method','Bayes','Rule','Soft','Noise','LevelIndependent', ...
    'DetrendMode','linear');

```

## batch_plot_six_columns.m

Run and choose a .txt, then Wavelet Signal Denoiser(Matlab) will be created based on each column.

## view_six_signals_from_mat.m

View signals from .mat(six columns).

## generate_event_config.py

Scans your data folder for `*_denoise.txt`, auto-fills an editable `events_config.json` (event type, per-event duration, start times, total duration). It can parse filenames like `Throat- Cough x6 every 10s_denoise.txt` or `喉咙-吞咽每5秒一次共31秒_denoise.txt`.

**Basic usage**

```bash
python generate_event_config.py --data-dir ./denoised_hyx_data --out ./events_config.json
```

**Key options**

- `--start-at-zero` — Start events at `0` then every Δ seconds: `[0, Δ, 2Δ, ...]`.
   Default starts at Δ: `[Δ, 2Δ, ...]`.
- `--default-label chew|drink|cough|swallow|talk|other` — Fallback label if not recognized from the filename.
- `--default-duration 2.0` — Default per-event duration (seconds) if not present in the filename.
- Incremental updates: if `events_config.json` already exists, the script **reuses** it and **appends only new** `_denoise` files (existing entries aren’t touched).

**Filename parsing (auto-fill)**

- **Count**: `共6次`, `6次`, `6x`
- **Interval**: `间隔10秒`, `每隔5秒`, `每5秒(一次)`, `2.5s`
- **Per-event duration (optional)**: `每次1秒`, `持续2s`
- **Total duration (optional)**: `共31秒`, `总时长31秒`, `31s total`
- Logic:
  - If **interval Δ + total T** → starts = `[Δ, 2Δ, ... ⌊T/Δ⌋·Δ]`, total = `T`
  - Else if **count N + interval Δ** → starts = `[Δ, 2Δ, ..., NΔ]`, total = `N·Δ`
  - Add `--start-at-zero` to instead get `[0, Δ, 2Δ, ...]`

**After running:**
 Open `events_config.json`, review and tweak `label`, `event_duration_sec`, `event_starts_sec`, `total_duration_sec` per file if needed.

 ## run_behavior_classification.py

### Train (JSON-only windowing, multimodal)

Basic long-ish run on **ear-canal only**:

```bat
python run_behavior_classification.py ^
  --data-dir .\denoised_hyx_data ^
  --event-config .\events_config.json ^
  --epochs 80 --batch-size 128 --lr 3e-4
```

Evaluate on **all** windows (huge confusion matrix):

```bat
python run_behavior_classification.py ^
  --data-dir .\denoised_hyx_data ^
  --event-config .\events_config.json ^
  --epochs 200 --batch-size 256 --emb-dim 320 --nlayers 8 --nhead 8 ^
  --eval-set all
```

10-fold cross-validation (recommended for reliable metrics):

```bat
python run_behavior_classification.py ^
  --data-dir .\denoised_hyx_data ^
  --event-config .\events_config.json ^
  --cv-folds 10 --epochs 80 --batch-size 192
```

> Add `--use-throat` to include throat files too.

### Results:

- Console INFO for each window:
   `start/duration -> [row_start:row_end) Hz=...`
- Outputs under `._prepared_events\`:
  - `windows\sample_*.npy`, `all_labels.json`
  - `train\labels.json`, `val\labels.json`
  - `channel_stats.json`
  - `_simple_outputs\training_accuracy.png`
  - `_simple_outputs\confusion_matrix.png` + `.json`
  - `_simple_outputs\training_metrics.json`
  - `_simple_outputs\best_model.pt`
  - (CV) `_simple_outputs\confusion_matrix_cv.png/json`, `cv_metrics.json`

## ignore_large_files.py

Run it directly to scan the current folder with a 100MB threshold and append oversized files to `.gitignore`.

```bash
python ignore_large_files.py
```

- Specify a scan root (e.g., your repo path).

```bash
python ignore_large_files.py --root /path/to/repo
```

- Customize the threshold with units like `MB/GiB` (e.g., 750MiB, 1GiB).

```bash
python ignore_large_files.py --threshold 750MiB
```

- Point to a specific `.gitignore` file (useful if it’s not in the root).

```bash
python ignore_large_files.py --gitignore /path/to/.gitignore
```

- Note: For files already tracked by Git, untrack them manually or `.gitignore` won’t take effect.

```bash
git rm --cached <file>    # then commit
```

---

**archived**

## ReadDirectory.py

Batch-process all `.txt` files inside a directory.

Make sure you have created a directory like `xxx_data/`, then run:

```bat
# Basic usage
python ReadDirectory.py --dir ./xxx_data --fs 100 --mains 50

# Specify PPG channel and use stronger suppression
python ReadDirectory.py --dir ./xxx_data --channel ir --tplN 400 --prom 1.0 --mu 1e-3 --order 16

# Disable the pre-ANC band-pass filter (for comparison)
python ReadDirectory.py --dir ./xxx_data --prebp none

```

This script will read all the `txt` files in the directory, and create to the result folder:

- A `.png` figure for each `.txt` file.
- A `summary.csv` that records parameters, number of peaks, ANC-before/after RMS/STD/PTP values, IMU energy, and other key metrics.

## Readtxt.py

Edit `txt = './xxx_data/xxx.txt'` in `Readtxt.py`, then run:

```bat
python ./Readtxt.py
```

## run_export_approx.m

Run this script in Matlab to denoise signal by wavelet approximation and export approximated signal in txt.

Tips: batch_export_approx_txt.m needed.

## approx_txt_viewer.py

```bat
python approx_txt_viewer.py
```

Scan and generate the approximated chart. 

## ppg_advanced_analysis.py

Analysis PPG signals via applying multiple denoising methods, performing frequency analysis with STFT, and generating comprehensive visualization charts.

Install dependences first:

```bat
python -m pip install numpy matplotlib scipy
```

Then run:

```bat
# 1. Batch process all *_data directories (default mode)
python ppg_advanced_analysis.py

# 2. Specify data directory pattern and channels
python ppg_advanced_analysis.py --batch '*_data' 'green,ir,red'

# 3. Specify output directory
python ppg_advanced_analysis.py --batch '*_data' 'green' './my_results'

# 4. Single file processing
python ppg_advanced_analysis.py single_file.txt green
```

## run_behavior_classification.py

**Dependences**: Windows, Anaconda, and Python3.11 environment.

```python
# check_cuda.py

import torch
print("torch:", torch.__version__)
print("built cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
```

```bat
conda create --name py311 python=3.11

conda activate py311

pip uninstall -y torch torchvision torchaudio

# Depend on nvidia-smi
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129

python ./check_cuda.py

pip install -U pandas numpy scipy scikit-learn matplotlib seaborn tqdm pyyaml joblib psutil pynvml ipykernel
```

It should output:

> torch: 2.8.0+cu129
> built cuda: 12.9
> cuda available: True
> device: NVIDIA GeForce RTX 4080 SUPER

Then run:

```bat
# Test GPU only
python run_behavior_classification.py --test-gpu

# Run full pipeline with mixed precision (recommended)
python run_behavior_classification.py --full-pipeline --mixed-precision --batch-size 64 --gpu-id 0

# Merge data via glob and run
python run_behavior_classification.py --full-pipeline --data-glob "./*_data" --merge-strategy copy
python run_behavior_classification.py --full-pipeline --data-glob "./ppg_denoised_data/*_de'no'i" --merge-strategy copy
```

## archived/

Saving all the archived files and code.

## Log

- 20250917-01 Added `ppg_advanced_analysis.py`.

- 20250824-02 Added `ReadDirectory.py`, which enables batch processing of all data files in a folder and generates figures. - tsawke

- 20250824-01 Replaced the MATLAB preprocessing step in the workflow with Python code (you can still use `Preprocess.m` to obtain `.mat` files and process them via `Readmat.py`).  - tsawke

  