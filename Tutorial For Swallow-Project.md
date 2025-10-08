# Tutorial For Project-Swallow

Github: [Project-Swallow](https://github.com/tsawke/Project-Swallow.git)

Latest package: [URL for Synology NAS](https://gofile.me/7xDmN/6LaSVzQNf)

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

  