# Tutorial For Swallow-Project

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


## archived/

Saving all the archived files and code.

## Log

- 20250824-02 Added `ReadDirectory.py`, which enables batch processing of all data files in a folder and generates figures. - tsawke

- 20250824-01 Replaced the MATLAB preprocessing step in the workflow with Python code (you can still use `Preprocess.m` to obtain `.mat` files and process them via `Readmat.py`).  - tsawke

  