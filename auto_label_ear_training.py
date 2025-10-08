#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
auto_label_ear_training.py
DENOISED INPUTS ONLY | FS-AUTO | GRID-ASSISTED + TOP-K | EVENT TIMESTAMPS | VERBOSE [INFO]

Highlights:
- NO wavelet/Wiener: your inputs are already denoised. We only do detrend/standardize (and optional band-pass emphasis).
- Auto estimate sampling rate (fs) per file from filename priors + row count; fallback to 50 Hz.
- Robust detection on THROAT: multi-feature fusion (IR envelope + ACC RMS + ACC jerk) + adaptive threshold +
  grid-assisted recovery + Top-K with spacing guided by priors.
- Prints event times (seconds) for each file in BOTH `--detect-only` and dataset building.
- Fixed SciPy detrend axis bug (use axis=-1 for 1D).
- Training: Conv+Transformer; AMP via torch.amp; GPU-aware data loading; full [INFO] logs; best-val metrics.

Usage:
  # 1) Detection sanity (timestamps included)
  python auto_label_ear_training.py --data-root ./denoised_hyx_data --detect-only

  # 2) Train (prints per-epoch and final best val acc + confusion matrix + per-class P/R/F1)
  python auto_label_ear_training.py --data-root ./denoised_hyx_data --epochs 30 --mixed-precision --batch-size 64
"""

import os, re, glob, argparse, warnings, time
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# ---- Silence only the specific pynvml deprecation spam from torch.cuda ----
warnings.filterwarnings(
    "ignore",
    message=r".*pynvml package is deprecated.*",
    category=FutureWarning,
    module=r"torch\.cuda"
)

# ---------------- Optional Torch (required for training/inference) ----------------
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_OK = True
except Exception as e:
    TORCH_OK = False
    print(f"[WARN] PyTorch not ready: {e}")

import scipy.signal as sg

# ---------------- Label mapping (display names in BASE_CLASS_NAME) ----------------
LABEL_MAP = {
    "静息": 0, "rest": 0,
    "咀嚼": 1, "chew": 1,
    "咳嗽": 2, "cough": 2,
    "吞咽": 3, "swallow": 3,
    "喝水": 3, "drink": 3,   # drink -> swallow
    "说话": 4, "讲话": 4, "说": 4, "speaking": 4, "speak": 4, "talk": 4, "talking": 4,
}
BASE_CLASS_NAME = {0: "rest", 1: "chew", 2: "cough", 3: "swallow", 4: "speak", 9999: "other"}

def infer_label_id_from_name(filename: str) -> Optional[int]:
    low = filename.lower()
    for k, v in LABEL_MAP.items():
        if k in low:
            return v
    return None

# ---------------- IO / Parsing ----------------
def read_approx_txt(path: str) -> np.ndarray:
    """Read *_approx.txt (denoised upstream): skip 2 header lines; expect >=6 numeric cols; return (N,6)."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    data = []
    for ln in lines[2:]:
        ln = ln.strip()
        if not ln:
            continue
        parts = re.split(r"[,\t ]+", ln)
        if len(parts) < 6:
            continue
        try:
            row = [float(x) for x in parts[:6]]
            data.append(row)
        except:
            continue
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 6:
        raise ValueError(f"Bad approx file: {path}")
    return arr[:, :6]

def pair_ear_throat(files: List[str]) -> Dict[str, Dict[str, str]]:
    """Pair files differing only by the token '耳道' vs '喉咙' in name."""
    pairs = {}
    for p in files:
        bn = Path(p).name
        if "耳道" in bn:
            key = bn.replace("耳道", "XXX")
            pairs.setdefault(key, {})["ear"] = p
        elif "喉咙" in bn:
            key = bn.replace("喉咙", "XXX")
            pairs.setdefault(key, {})["throat"] = p
    pairs = {k: v for k, v in pairs.items() if "ear" in v and "throat" in v}
    print(f"[INFO] Found {len(pairs)} ear/throat pairs.")
    return pairs

# ---------------- Priors from filename ----------------
PRIORS_COUNT_PAT = re.compile(r"(?:共)?(\d+)\s*次")
PRIORS_IV_PAT    = re.compile(r"间隔\s*(\d+)\s*秒")
PRIORS_DUR_PAT   = re.compile(r"持续\s*(\d+)\s*秒")

def parse_priors_from_name(name: str):
    """Return (K_expected, interval_s, duration_s) if present."""
    K = Iv = Dur = None
    m = PRIORS_COUNT_PAT.search(name)
    if m: K = int(m.group(1))
    m = PRIORS_IV_PAT.search(name)
    if m: Iv = int(m.group(1))
    m = PRIORS_DUR_PAT.search(name)
    if m: Dur = int(m.group(1))
    return K, Iv, Dur

def estimate_fs_from_length_and_priors(N: int, K: Optional[int], Iv: Optional[int],
                                       Dur: Optional[int], fallback: int = 50) -> int:
    """Estimate fs using filename priors (K/Iv/Dur) + sample count N; snap to common rates; fallback=50."""
    if K and Iv:
        T = (max(1, K - 1)) * Iv + (K * Dur if Dur else 0)
        if T > 0:
            fs_est = N / T
            for cand in [40, 50, 60, 80, 100, 128, 200]:
                if abs(fs_est - cand) < cand * 0.15:
                    return cand
            return max(25, int(round(fs_est)))
    return fallback

# ---------------- Detection utilities (THROAT) ----------------
def detrend_if(x, axis=None):
    """Detrend helper: use axis=-1 for 1D arrays to avoid SciPy TypeError."""
    if axis is None:
        return sg.detrend(x, axis=-1)  # fix for 1D arrays
    return sg.detrend(x, axis=axis)

def robust_z1d(x, eps=1e-9):
    x = np.asarray(x)
    med = np.median(x)
    iqr = np.percentile(x, 75) - np.percentile(x, 25)
    return (x - med) / (iqr + eps)

def moving_rms(x, win):
    if win <= 1:
        return np.abs(x)
    k = np.ones(win) / win
    return np.sqrt(sg.convolve(x**2, k, mode="same"))

def butter_bandpass(x, fs, low, high, order=4):
    b, a = sg.butter(order, [low / (fs / 2), high / (fs / 2)], btype="band")
    return sg.filtfilt(b, a, x, axis=0) if x.ndim > 1 else sg.filtfilt(b, a, x)

def choose_topk_with_spacing(peaks, score, K, fs, interval_s=None):
    if len(peaks) == 0 or not K or K <= 0:
        return np.asarray(peaks, dtype=int)
    order = np.argsort(score[peaks])[::-1]
    selected = []
    min_dist = int((interval_s * 0.4) * fs) if interval_s else 0  # allow overlaps if needed
    for idx in order:
        p = peaks[idx]
        if min_dist > 0 and any(abs(p - q) < min_dist for q in selected):
            continue
        selected.append(p)
        if len(selected) >= K:
            break
    if len(selected) < K:
        for idx in order:
            p = peaks[idx]
            if p not in selected:
                selected.append(p)
                if len(selected) >= K:
                    break
    return np.array(sorted(selected), dtype=int)

def grid_assisted_peaks(fuse: np.ndarray, K: int, fs: int, interval_s: Optional[int]):
    """Split by Iv*fs (if Iv given) or equally into K segments; take max per segment."""
    N = len(fuse)
    if not K or K <= 0 or N == 0:
        return np.array([], dtype=int)
    seg_len = int(interval_s * fs) if interval_s else (N // K if K > 0 else N)
    seg_len = max(1, seg_len)
    peaks = []
    for j in range(K):
        s = j * seg_len
        e = min(N, (j + 1) * seg_len)
        if e - s < 5:
            continue
        idx = s + int(np.argmax(fuse[s:e]))
        peaks.append(idx)
    return np.array(sorted(set(peaks)), dtype=int)

def detect_events_from_throat(throat: np.ndarray, fs: int,
                              use_bandpass: bool = True,
                              lam: float = 1.6,
                              global_edge_trim_s: float = 0.5,
                              min_dist_s: float = 0.8,
                              K_expected: Optional[int] = None,
                              interval_s: Optional[int] = None,
                              duration_s: Optional[int] = None):
    """
    Multi-feature fusion + adaptive threshold + grid-assisted recovery + Top-K (guided by priors).
    Returns:
        peaks (np.ndarray[int]): sample indices of detected events on THROAT timeline.
        fuse  (np.ndarray[float]): fused detection score track.
        thr   (float): last threshold used.
    """
    n = throat.shape[0]
    ir = throat[:, 1]          # IR PPG
    acc = throat[:, 3:6]       # ACC x,y,z

    # detrend (NOT denoising)
    ir = detrend_if(ir)
    acc = detrend_if(acc, axis=0)

    # emphasis band (default ON, highlights target band; still not "denoising")
    if use_bandpass:
        ir = butter_bandpass(ir, fs, 0.4, 8.0)
        acc = butter_bandpass(acc, fs, 0.5, 8.0)

    # features
    ir_env   = np.abs(sg.hilbert(ir))
    acc_rms  = moving_rms(np.linalg.norm(acc, axis=1), win=max(2, int(0.04 * fs)))
    acc_jerk = np.linalg.norm(np.diff(acc, axis=0, prepend=acc[:1]), axis=1)

    # normalize & fuse
    z_ir   = robust_z1d(ir_env)
    z_rms  = robust_z1d(acc_rms)
    z_jerk = robust_z1d(acc_jerk)
    fuse = 0.5 * z_ir + 0.35 * z_rms + 0.15 * z_jerk

    # trim edges (suppress start/end artifacts)
    trim = int(global_edge_trim_s * fs)
    if trim > 0 and n > 2 * trim:
        base_front = np.median(fuse[trim:min(n, trim + fs)])
        base_back  = np.median(fuse[max(0, n - trim - fs):max(0, n - trim)])
        fuse[:trim] = base_front
        fuse[-trim:] = base_back

    # adaptive threshold search
    thr = np.mean(fuse) + lam * np.std(fuse)
    peaks, props = sg.find_peaks(fuse, height=thr, distance=int(min_dist_s * fs), prominence=0.3 * np.std(fuse))
    lam_curr = lam
    need = int(0.8 * K_expected) if K_expected else 0
    tries = 0
    while K_expected and len(peaks) < max(2, need) and tries < 4:
        lam_curr -= 0.2
        thr = np.mean(fuse) + lam_curr * np.std(fuse)
        peaks, props = sg.find_peaks(fuse, height=thr, distance=int(min_dist_s * fs), prominence=0.2 * np.std(fuse))
        tries += 1

    # Grid-assisted if still few
    if K_expected and len(peaks) < need:
        grid = grid_assisted_peaks(fuse, K_expected, fs, interval_s)
        if grid.size > 0:
            peaks = np.unique(np.concatenate([peaks, grid]))

    # Final Top-K with spacing by interval
    if K_expected:
        peaks = choose_topk_with_spacing(peaks, fuse, K_expected, fs, interval_s=interval_s)

    return peaks, fuse, thr

# ---------------- Pretty print times ----------------
def format_times(peaks: np.ndarray, fs: int) -> str:
    if peaks is None or len(peaks) == 0:
        return "(none)"
    return ", ".join([f"{p / fs:.2f}s" for p in peaks])

# ---------------- Dataset (resample to fixed length) ----------------
def resample_fixed_length(x: np.ndarray, target_len: int) -> np.ndarray:
    """Resample each of 6 channels to target_len along time axis; x:(T,6) -> (target_len,6)."""
    T = x.shape[0]
    if T == target_len:
        return x
    return np.stack([sg.resample(x[:, i], target_len) for i in range(x.shape[1])], axis=1)

class EarEventDataset(Dataset):
    """Windows on EAR, fixed length after resampling. No denoising."""
    def __init__(self, ear_segments: List[np.ndarray], labels: List[int], fixed_len: int,
                 detrend: bool = True, zscore: bool = True):
        self.X = []
        self.y = labels
        self.fixed_len = fixed_len
        for seg in ear_segments:
            seg = resample_fixed_length(seg, fixed_len)
            if detrend:
                seg = sg.detrend(seg, axis=0)
            if zscore:
                mu = seg.mean(0, keepdims=True)
                sd = seg.std(0, keepdims=True) + 1e-6
                seg = (seg - mu) / sd
            self.X.append(torch.from_numpy(seg.T.astype(np.float32)))  # (C,T)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], torch.tensor(self.y[i], dtype=torch.long)

# ---------------- Model ----------------
class ConvTransformer(nn.Module):
    def __init__(self, in_ch=6, num_classes=4, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, 64, 7, padding=3), nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 5, padding=2),   nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        self.proj = nn.Conv1d(128, d_model, 1)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.cls = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(d_model, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(128, num_classes),
        )
    def forward(self, x):          # x: (B,C,T)
        x = self.conv(x)           # (B,128,T/2)
        x = self.proj(x)           # (B,d_model,T/2)
        x = x.transpose(1, 2)      # (B,T/2,d_model)
        x = self.enc(x)            # (B,T/2,d_model)
        x = x.transpose(1, 2)      # (B,d_model,T/2)
        return self.cls(x)         # (B,num_classes)

# ---------------- Train / Eval ----------------
def setup_gpu(allow_cpu=False, mixed_precision=False, compile_try=True):
    use_cuda = torch.cuda.is_available() and not allow_cpu
    device = torch.device("cuda" if use_cuda else "cpu")
    amp = bool(mixed_precision and device.type == "cuda")
    if use_cuda:
        print(f"[INFO] CUDA available. device={torch.cuda.get_device_name(0)}")
    else:
        print("[INFO] CUDA not available -> run on CPU.")
    def maybe_compile(m):
        if compile_try and hasattr(torch, "compile"):
            try:
                m2 = torch.compile(m); print("[INFO] torch.compile enabled.")
                return m2
            except Exception as e:
                print(f"[WARN] torch.compile failed: {e}")
                return m
        return m
    return device, amp, maybe_compile, use_cuda

@torch.no_grad()
def evaluate_and_cm(model, dl, device, K=None):
    model.eval(); correct = 0; n = 0
    cm = np.zeros((K, K), dtype=int) if K else None
    for xb, yb in dl:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb).argmax(1)
        correct += (pred == yb).sum().item(); n += yb.size(0)
        if cm is not None:
            for t, p in zip(yb.cpu().numpy(), pred.cpu().numpy()):
                cm[t, p] += 1
    return (correct / max(1, n)), cm

def print_confusion_and_metrics(cm: np.ndarray, class_names: List[str]):
    print("[INFO] Validation Confusion Matrix (rows=true, cols=pred):")
    header = "          " + " ".join([f"{n[:6]:>6}" for n in class_names])
    print(header)
    for i, row in enumerate(cm):
        print(f"{class_names[i][:8]:>8} " + " ".join([f"{v:6d}" for v in row]))
    print("[INFO] Per-class metrics:")
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        support = cm[i, :].sum()
        print(f"[INFO] {name:>8} | P={prec:.3f} R={rec:.3f} F1={f1:.3f} | support={support}")

def train_loop(model, dl_tr, dl_va, device, amp, epochs=30, lr=1e-3, grad_acc=1, class_names=None):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    best = {"acc": -1, "state": None, "cm": None}
    ce = nn.CrossEntropyLoss()
    print("[INFO] Start training...")
    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train(); loss_sum = 0.0; batch_count = 0; correct = 0; n = 0
        opt.zero_grad(set_to_none=True)
        for i, (xb, yb) in enumerate(dl_tr):
            xb, yb = xb.to(device), yb.to(device)
            with torch.amp.autocast(device_type='cuda', enabled=amp):
                logits = model(xb)
                loss = ce(logits, yb) / grad_acc
            scaler.scale(loss).backward()
            if (i + 1) % grad_acc == 0:
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            loss_sum += loss.item() * grad_acc; batch_count += 1
            correct += (logits.argmax(1) == yb).sum().item(); n += yb.size(0)
        if batch_count == 0:
            raise RuntimeError("No training batches produced; lower --batch-size or adjust detection.")
        tr_acc = correct / max(1, n)
        va_acc, cm = evaluate_and_cm(model, dl_va, device, K=len(class_names) if class_names else None)
        dt = time.time() - t0
        print(f"[INFO] Epoch {ep:02d} | loss={loss_sum / max(1, batch_count):.4f} | "
              f"train_acc={tr_acc:.3f} | val_acc={va_acc:.3f} | {dt:.1f}s")
        if va_acc > best["acc"]:
            best = {"acc": va_acc, "state": model.state_dict(), "cm": cm}
    print("[INFO] Training finished.")
    if best["state"] is not None:
        model.load_state_dict(best["state"])
        print(f"[INFO] Best val_acc={best['acc']:.3f}")
        if class_names and best["cm"] is not None:
            print_confusion_and_metrics(best["cm"], class_names)
    return model

# ---------------- Windowing on EAR ----------------
def extract_ear_windows(ear: np.ndarray, centers: np.ndarray, fs: int,
                        w_pre=0.5, w_post=1.5, local_trim=0.1) -> List[np.ndarray]:
    """Cut windows around THROAT event centers but on EAR signal; trim inner edges; no denoising."""
    N = ear.shape[0]; out = []
    pre = int(w_pre * fs); post = int(w_post * fs); cut = int(local_trim * fs)
    for c in centers:
        s = max(0, c - pre); e = min(N, c + post)
        seg = ear[s:e, :]
        if seg.shape[0] <= 2 * cut: 
            continue
        seg = seg[cut:-cut, :]
        out.append(seg)
    return out

# ---------------- Build dataset (with timestamp prints) ----------------
def build_dataset_from_pairs(pairs: Dict[str, Dict[str, str]],
                             target_fs_win: int = 100,
                             w_pre: float = 0.5, w_post: float = 1.5, local_trim: float = 0.1,
                             lam: float = 1.6, use_bandpass: bool = True):
    ear_segments = []; labels = []; class_names = []
    print("[INFO] Start event detection on THROAT and windowing on EAR...")
    for key, d in pairs.items():
        ear_file, thr_file = d["ear"], d["throat"]
        name = Path(ear_file).name
        Kexp, Iexp, Dexp = parse_priors_from_name(name)
        lbl = infer_label_id_from_name(name)
        if lbl is None:
            print(f"[INFO] {name} -> class=other")
            lbl = 9999

        ear = read_approx_txt(ear_file)
        thr = read_approx_txt(thr_file)
        fs_est = estimate_fs_from_length_and_priors(len(thr), Kexp, Iexp, Dexp, fallback=50)

        peaks, fuse, thr_value = detect_events_from_throat(
            thr, fs=fs_est, use_bandpass=use_bandpass, lam=lam,
            global_edge_trim_s=0.5, min_dist_s=0.8,
            K_expected=Kexp, interval_s=Iexp, duration_s=Dexp
        )
        # Print timestamps here
        print(f"[INFO] {name}: priors(K={Kexp},Iv={Iexp}s,Dur={Dexp}s) fs_est={fs_est}Hz -> "
              f"detected {len(peaks)} event(s) at [{format_times(peaks, fs_est)}].")

        ear_wins = extract_ear_windows(ear, peaks, fs=fs_est, w_pre=w_pre, w_post=w_post, local_trim=local_trim)
        if len(ear_wins) == 0:
            print(f"[WARN] {name}: no EAR windows extracted; check params.")
            continue

        ear_segments += ear_wins
        labels += [lbl] * len(ear_wins)

    if len(ear_segments) == 0:
        print("[ERROR] No samples created; check detection threshold/priors.")
        return [], [], [], 0

    # Reindex labels to contiguous 0..K-1
    uniq = sorted(set(labels))
    remap = {old: i for i, old in enumerate(uniq)}
    labels = [remap[x] for x in labels]
    for old in uniq:
        class_names.append(BASE_CLASS_NAME.get(old, "other") if old != 9999 else "other")

    # Dataset summary
    print(f"[INFO] Dataset built: samples={len(ear_segments)} classes={len(class_names)} -> {class_names}")
    counts = np.zeros(len(class_names), dtype=int)
    for y in labels: counts[y] += 1
    for i, nm in enumerate(class_names):
        print(f"[INFO] Class '{nm}' samples={counts[i]}")

    fixed_len = int(round((w_pre + w_post - 2 * local_trim) * target_fs_win))
    fixed_len = max(10, fixed_len)
    return ear_segments, labels, class_names, fixed_len

# ---------------- Sliding inference helper (optional) ----------------
@torch.no_grad()
def sliding_infer(model, ear: np.ndarray, fs: int, device, win_s=2.0, hop_s=0.25, amp=False, target_fs_win: int = 100):
    """Optional util; not used by training path. Kept for debugging."""
    T = ear.shape[0]; W = int(win_s * fs); H = int(hop_s * fs); outs = []
    i = 0
    while i + W <= T:
        seg = ear[i:i + W, :]
        seg = sg.detrend(seg, axis=0)
        seg = resample_fixed_length(seg, int(round(win_s * target_fs_win)))
        mu = seg.mean(0, keepdims=True); sd = seg.std(0, keepdims=True) + 1e-6
        seg = (seg - mu) / sd
        xb = torch.from_numpy(seg.T.astype(np.float32)).unsqueeze(0).to(device)
        with torch.amp.autocast(device_type='cuda', enabled=amp and device.type == "cuda"):
            prob = F.softmax(model(xb), dim=1).squeeze(0).cpu().numpy()
        outs.append((i, i + W, prob.argmax(), prob))
        i += H
    print(f"[INFO] Sliding inference produced {len(outs)} windows.")
    return outs

# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--mixed-precision", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--grad-acc", type=int, default=1)

    # detection config
    ap.add_argument("--lam", type=float, default=1.6, help="threshold lambda (mean + lam*std)")
    ap.add_argument("--no-bandpass-detect", action="store_true", help="disable band-pass emphasis on THROAT")
    ap.add_argument("--detect-only", action="store_true")

    # window / resample config
    ap.add_argument("--w-pre", type=float, default=0.5)
    ap.add_argument("--w-post", type=float, default=1.5)
    ap.add_argument("--local-trim", type=float, default=0.1)
    ap.add_argument("--target-fs-win", type=int, default=100, help="resample EAR windows to this fs for fixed length")

    args = ap.parse_args()

    # Scan & pair
    files = glob.glob(os.path.join(args.data_root, "*approx.txt"))
    print(f"[INFO] Scanning approx files under {args.data_root} -> {len(files)}")
    if not files:
        raise SystemExit("[ERROR] No approx files.")
    pairs = pair_ear_throat(files)
    if not pairs:
        raise SystemExit("[ERROR] No ear/throat pairs found.")

    # DETECT-ONLY branch (prints timestamps!)
    if args.detect_only:
        for key, d in pairs.items():
            ear_file, thr_file = d["ear"], d["throat"]
            name = Path(ear_file).name
            Kexp, Iexp, Dexp = parse_priors_from_name(name)
            thr = read_approx_txt(thr_file)
            fs_est = estimate_fs_from_length_and_priors(len(thr), Kexp, Iexp, Dexp, fallback=50)
            peaks, fuse, thrv = detect_events_from_throat(
                thr, fs=fs_est, use_bandpass=(not args.no_bandpass_detect), lam=args.lam,
                global_edge_trim_s=0.5, min_dist_s=0.8,
                K_expected=Kexp, interval_s=Iexp, duration_s=Dexp
            )
            print(f"[INFO] {name}: priors(K={Kexp},Iv={Iexp},Dur={Dexp}) fs_est={fs_est}Hz -> "
                  f"detected {len(peaks)} event(s) at [{format_times(peaks, fs_est)}].")
        print("[INFO] Done (--detect-only).")
        return

    # TRAIN branch
    if not TORCH_OK:
        raise SystemExit("[ERROR] PyTorch required for training.")

    ear_segments, labels, class_names, fixed_len = build_dataset_from_pairs(
        pairs, target_fs_win=args.target_fs_win,
        w_pre=args.w_pre, w_post=args.w_post, local_trim=args.local_trim,
        lam=args.lam, use_bandpass=(not args.no_bandpass_detect)
    )
    if len(ear_segments) == 0:
        raise SystemExit("[ERROR] No samples produced; tune --lam or ensure filename priors exist.")

    # Split
    idx = np.arange(len(ear_segments)); np.random.shuffle(idx)
    split = int(0.8 * len(idx))
    tr_idx, va_idx = idx[:split], idx[split:]
    Xtr = [ear_segments[i] for i in tr_idx]; ytr = [labels[i] for i in tr_idx]
    Xva = [ear_segments[i] for i in va_idx]; yva = [labels[i] for i in va_idx]
    print(f"[INFO] Train/Val split -> train={len(Xtr)} val={len(Xva)} fixed_len={fixed_len}")

    ds_tr = EarEventDataset(Xtr, ytr, fixed_len=fixed_len, detrend=True, zscore=True)
    ds_va = EarEventDataset(Xva, yva, fixed_len=fixed_len, detrend=True, zscore=True)

    # Loaders
    bs = min(args.batch_size, max(1, len(ds_tr)))
    device, amp, maybe_compile, use_cuda = setup_gpu(allow_cpu=args.allow_cpu,
                                                     mixed_precision=args.mixed_precision,
                                                     compile_try=args.compile)
    pin_mem = bool(use_cuda)
    nw = min(os.cpu_count() or 2, 8)
    dl_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True,
                       num_workers=nw, pin_memory=pin_mem, persistent_workers=bool(nw > 0), drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=max(1, min(bs, len(ds_va))), shuffle=False,
                       num_workers=max(1, nw // 2), pin_memory=pin_mem, persistent_workers=bool(nw > 0), drop_last=False)
    print(f"[INFO] DataLoader -> workers(train,val)=({nw},{max(1, nw//2)}) pin_memory={pin_mem} bs={bs}")

    # Classes
    num_classes = len(set(labels))
    print(f"[INFO] Classes detected: {num_classes} -> {class_names}")

    model = ConvTransformer(in_ch=6, num_classes=num_classes)
    model = maybe_compile(model).to(device)

    model = train_loop(model, dl_tr, dl_va, device, amp,
                       epochs=args.epochs, lr=args.lr, grad_acc=args.grad_acc, class_names=class_names)

    os.makedirs("artifacts", exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "class_names": class_names}, "artifacts/ear_conv_transformer.pt")
    print("[INFO] Saved checkpoint to artifacts/ear_conv_transformer.pt")
    print("[INFO] All done.")

if __name__ == "__main__":
    main()
