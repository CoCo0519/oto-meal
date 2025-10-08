# -*- coding: utf-8 -*-
"""
single_file_event_annotator.py
Standalone annotator for one EAR waveform txt (6 channels, *_approx.txt).
- Loads trained model artifacts/ear_conv_transformer.pt
- Sliding-window inference (window=2.0s, hop=0.25s); resample window to 200 pts (100Hz*2s) to match training
- Conservative eventization to reduce false positives
- Visualize with correct time axis (seconds): shaded regions + text labels
- Export CSV and PNG; optional --show to pop up a window

Usage (PowerShell):
  python single_file_event_annotator.py --ear-file .\denoised_hyx_data\耳道-吞咽6次间隔10秒_approx.txt --use-priors --show
  # If events too many, be stricter:
  python single_file_event_annotator.py --ear-file .\...\耳道-咳嗽共6次间隔10秒_approx.txt --use-priors --peak 0.7 --alpha 3.5 --min-dur 1.0
  # If too few, be looser:
  python single_file_event_annotator.py --ear-file .\...\耳道-说话共6次持续5秒间隔10秒_approx.txt --peak 0.55 --alpha 2.5

NOTE: We DO NOT denoise again (your data already denoised). Only detrend + z-score.
"""

import os, re, argparse, warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.signal as sg
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# --- Windows-friendly: suppress torch cuda pynvml spam & set Chinese fonts ---
warnings.filterwarnings("ignore", message=r".*pynvml package is deprecated.*", category=FutureWarning, module=r"torch\.cuda")
matplotlib.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
matplotlib.rcParams["axes.unicode_minus"] = False

# ---------- Priors & label helpers ----------
PRIORS_COUNT_PAT = re.compile(r"(?:共)?(\d+)\s*次")
PRIORS_IV_PAT    = re.compile(r"间隔\s*(\d+)\s*秒")
PRIORS_DUR_PAT   = re.compile(r"持续\s*(\d+)\s*秒")

FNAME2CLASS = {
    "静息": "rest", "rest": "rest",
    "咀嚼": "chew", "chew": "chew",
    "咳嗽": "cough", "cough": "cough",
    "吞咽": "swallow", "喝水": "swallow", "swallow": "swallow", "drink": "swallow",
    "说话": "speak", "讲话": "speak", "speaking": "speak", "speak": "speak", "talk": "speak",
}

COLOR_MAP = {
    "swallow": (0.2, 0.6, 1.0, 0.25),  # translucent blue
    "chew":    (0.2, 1.0, 0.6, 0.25),  # greenish
    "cough":   (1.0, 0.4, 0.2, 0.25),  # orange-red
    "speak":   (0.7, 0.3, 1.0, 0.25),  # purple
    "rest":    (0.7, 0.7, 0.7, 0.15),
    "other":   (0.5, 0.5, 0.5, 0.15),
}

def infer_target_class_from_name(name: str, class_names):
    low = name.lower()
    for k, v in FNAME2CLASS.items():
        if k in low and v in class_names:
            return v
    for cn in class_names:
        if cn in low: return cn
    return None

def parse_priors_from_name(name:str):
    K = Iv = Dur = None
    m = PRIORS_COUNT_PAT.search(name);  K   = int(m.group(1)) if m else None
    m = PRIORS_IV_PAT.search(name);     Iv  = int(m.group(1)) if m else None
    m = PRIORS_DUR_PAT.search(name);    Dur = int(m.group(1)) if m else None
    return K, Iv, Dur

def estimate_fs_from_length_and_priors(N:int, K,Iv,Dur, fallback:int=50)->int:
    if K and Iv:
        T = (max(1,K-1))*Iv + (K*Dur if Dur else 0)
        if T>0:
            fs_est = N / T
            for cand in [40,50,60,80,100,128,200]:
                if abs(fs_est-cand) < cand*0.15: return cand
            return max(25, int(round(fs_est)))
    return fallback

# ---------- IO & preprocessing ----------
def read_approx_txt(path: str) -> np.ndarray:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    data=[]
    for ln in lines[2:]:
        ln=ln.strip()
        if not ln: continue
        parts = re.split(r"[,\t ]+", ln)
        if len(parts) < 6: continue
        try:
            row = [float(x) for x in parts[:6]]
            data.append(row)
        except: continue
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim!=2 or arr.shape[1]<6:
        raise ValueError(f"Bad approx file: {path}")
    return arr[:, :6]

def detrend_and_zscore(x: np.ndarray):
    x = sg.detrend(x, axis=0)  # NOT denoising
    mu = x.mean(0, keepdims=True); sd = x.std(0, keepdims=True)+1e-6
    return (x-mu)/sd

def resample_fixed_length(x: np.ndarray, target_len:int) -> np.ndarray:
    T = x.shape[0]
    if T == target_len: return x
    return np.stack([sg.resample(x[:,i], target_len) for i in range(x.shape[1])], axis=1)

# ---------- Model (must match training) ----------
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
    def forward(self, x):
        x = self.conv(x)           # (B,128,T/2)
        x = self.proj(x)           # (B,d_model,T/2)
        x = x.transpose(1,2)       # (B,T/2,d_model)
        x = self.enc(x)            # (B,T/2,d_model)
        x = x.transpose(1,2)
        return self.cls(x)         # (B,num_classes)

# ---------- Sliding inference ----------
@torch.no_grad()
def sliding_predict_proba(model, ear: np.ndarray, fs:int, device,
                          win_s=2.0, hop_s=0.25, target_fs_win:int=100):
    W=int(win_s*fs); H=int(hop_s*fs)
    T=ear.shape[0]
    starts=[]; probs=[]
    i=0
    model.eval()
    while i+W<=T:
        seg = ear[i:i+W,:].astype(np.float32)
        seg = resample_fixed_length(seg, int(round(win_s*target_fs_win)))  # keep same as training
        seg = detrend_and_zscore(seg)
        xb = torch.from_numpy(seg.T).unsqueeze(0).to(device)  # (1,C,T)
        out = F.softmax(model(xb), dim=1).squeeze(0).cpu().numpy()  # (C,)
        starts.append(i); probs.append(out)
        i += H
    starts = np.array(starts, dtype=int)
    probs  = np.stack(probs, axis=0) if len(probs)>0 else np.zeros((0,1))
    centers = starts + int(W/2)
    times_s = centers / fs
    return times_s, probs  # (Nwin,), (Nwin,C)

# ---------- Conservative eventization ----------
def mad(x):
    m = np.median(x)
    return np.median(np.abs(x-m)) + 1e-12

def connected_components(mask):
    runs=[]; in_run=False; s=0
    for i, v in enumerate(mask):
        if v and not in_run:
            in_run=True; s=i
        elif not v and in_run:
            runs.append((s,i)); in_run=False
    if in_run: runs.append((s,len(mask)))
    return runs

def conservative_eventization(times_s, probs, class_names,
                              peak=0.65, alpha=3.0, q=0.9,
                              min_dur_s=0.8, margin=0.20,
                              merge_within_s=1.0,
                              use_priors=False, K_prior=None, Iv_prior=None,
                              target_class=None):
    """
    Returns: final events [(time_s, class_name, prob)], and per-class candidates for logging.
    """
    events=[]
    cand_by_cls={}
    if probs.shape[0]==0: return events, cand_by_cls

    C=probs.shape[1]
    dt = float(np.median(np.diff(times_s))) if len(times_s)>1 else 0.25
    min_dur_pts = max(1, int(min_dur_s / max(dt,1e-6)))

    def smooth(y,k=3):
        if k<=1: return y
        ker = np.ones(min(k, len(y))) / max(1, min(k, len(y)))
        return np.convolve(y, ker, mode="same")

    # 1) per-class candidates by adaptive threshold + min duration + margin
    cands=[]
    for ci in range(C):
        cname = class_names[ci] if ci < len(class_names) else f"class{ci}"
        if cname in ("rest","other"):  # ignore background
            continue
        track = smooth(probs[:,ci], k=3)
        thr = max(peak, np.median(track)+alpha*mad(track), np.quantile(track, q))
        mask = track >= thr
        runs = connected_components(mask)
        cand=[]
        for (s,e) in runs:
            if (e-s) < min_dur_pts: continue
            j = s + int(np.argmax(track[s:e]))
            p = float(track[j])
            margin_val = float(p - np.max(probs[j, np.arange(C)!=ci])) if C>1 else p
            if margin_val < margin: continue
            t = float(times_s[j])
            cand.append((t, cname, p))
            cands.append((t, cname, p))
        cand_by_cls[cname]=cand

    if not cands:
        return [], cand_by_cls

    # 2) temporal merge (NMS-like)
    cands.sort(key=lambda x: x[0])
    merged=[cands[0]]
    for t,c,p in cands[1:]:
        t0,c0,p0 = merged[-1]
        if abs(t - t0) <= merge_within_s:
            # keep higher prob, tie -> prefer target class if set
            if (p > p0) or (abs(p-p0)<1e-6 and target_class and c==target_class and c0!=target_class):
                merged[-1] = (t,c,p)
        else:
            merged.append((t,c,p))

    # 3) apply priors (if requested): keep only target_class; Top-K with spacing ~ Iv
    if use_priors and target_class and K_prior:
        tgt = [(t,c,p) for (t,c,p) in merged if c==target_class]
        if not tgt:
            return [], cand_by_cls
        times = np.array([t for (t,_,_) in tgt], dtype=float)
        scores= np.array([p for (_,_,p) in tgt], dtype=float)
        order = np.argsort(scores)[::-1]
        keep=[]
        sep = Iv_prior*0.6 if Iv_prior else 1.0
        for k in order:
            if any(abs(times[k]-times[j]) < sep for j in keep):
                continue
            keep.append(k)
            if len(keep)>=K_prior: break
        keep = sorted(keep, key=lambda idx: times[idx])
        final = [(times[i], target_class, scores[i]) for i in keep[:K_prior]]
        return final, cand_by_cls

    return merged, cand_by_cls

# ---------- Visualization ----------
def plot_with_shaded_events(ear: np.ndarray, fs:int, events, png_path:str,
                            title:str="", shade_half_width_s:float=0.6, show=False):
    t = np.arange(len(ear))/fs
    y = ear[:,1]  # IR PPG
    plt.figure(figsize=(12,4))
    plt.plot(t, y, linewidth=1)
    ymax = np.max(y) if len(y)>0 else 1.0
    ymin = np.min(y) if len(y)>0 else -1.0
    for (ts, cname, prob) in events:
        c = COLOR_MAP.get(cname, (0.3,0.3,0.3,0.2))
        plt.axvspan(max(0, ts-shade_half_width_s), min(t[-1] if len(t)>0 else ts+shade_half_width_s, ts+shade_half_width_s),
                    facecolor=c, edgecolor=None)
        plt.text(ts, ymax, f"{cname}@{ts:.2f}s", rotation=90, fontsize=8, va="bottom", ha="center")
    plt.xlabel("Time (s)"); plt.ylabel("Amplitude (IR PPG)")
    if title: plt.title(title)
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    if show:
        plt.show()
    plt.close()

def save_csv(events, csv_path:str, file_name:str):
    import csv
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w=csv.writer(f)
        w.writerow(["file","time_s","class","prob"])
        for ts, cname, prob in events:
            w.writerow([file_name, f"{ts:.4f}", cname, f"{prob:.4f}"])

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ear-file", type=str, required=True, help="单个耳道 *_approx.txt 文件")
    ap.add_argument("--model", type=str, default="artifacts/ear_conv_transformer.pt")
    ap.add_argument("--fs", type=int, default=0, help="采样率；0=自动估计(建议保留0)")
    ap.add_argument("--win", type=float, default=2.0)
    ap.add_argument("--hop", type=float, default=0.25)
    ap.add_argument("--target-fs-win", type=int, default=100)

    # conservative eventization
    ap.add_argument("--peak", type=float, default=0.65)
    ap.add_argument("--alpha", type=float, default=3.0)
    ap.add_argument("--q", type=float, default=0.90)
    ap.add_argument("--min-dur", type=float, default=0.8)
    ap.add_argument("--margin", type=float, default=0.20)
    ap.add_argument("--merge-within", type=float, default=1.0)
    ap.add_argument("--use-priors", action="store_true", help="仅目标类别，按 K/Iv 做 Top-K+间隔约束")
    ap.add_argument("--shade-half", type=float, default=0.6, help="可视化色带半宽(秒)")

    ap.add_argument("--outdir", type=str, default="annotate_outputs")
    ap.add_argument("--show", action="store_true", help="显示窗口")
    ap.add_argument("--allow-cpu", action="store_true")
    args = ap.parse_args()

    # load model
    if not os.path.exists(args.model):
        raise SystemExit(f"[ERROR] model not found: {args.model}. Train first or point to your .pt.")
    ckpt = torch.load(args.model, map_location="cpu")
    class_names = ckpt.get("class_names", [f"class{i}" for i in range(5)])
    model = ConvTransformer(in_ch=6, num_classes=len(class_names))
    model.load_state_dict(ckpt["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() and not args.allow_cpu else "cpu")
    model.to(device)
    print(f"[INFO] Model loaded. classes={class_names} device={device.type}")

    # read file & estimate fs
    ear_path = args.ear_file
    if not os.path.exists(ear_path):
        raise SystemExit(f"[ERROR] file not found: {ear_path}")
    ear = read_approx_txt(ear_path)
    name = Path(ear_path).name
    K,Iv,Dur = parse_priors_from_name(name)
    fs = args.fs if args.fs>0 else estimate_fs_from_length_and_priors(len(ear), K,Iv,Dur, fallback=50)
    target_class = infer_target_class_from_name(name, class_names)
    print(f"[INFO] {name}: fs={fs}Hz priors(K={K},Iv={Iv},Dur={Dur}) target_class={target_class}")

    # sliding inference
    times, probs = sliding_predict_proba(model, ear, fs, device,
                                         win_s=args.win, hop_s=args.hop, target_fs_win=args.target_fs_win)
    print(f"[INFO] {name}: produced {len(times)} windows.")

    # eventization
    events, cand_by_cls = conservative_eventization(
        times, probs, class_names,
        peak=args.peak, alpha=args.alpha, q=args.q,
        min_dur_s=args.min_dur, margin=args.margin,
        merge_within_s=args.merge_within,
        use_priors=args.use_priors, K_prior=K, Iv_prior=Iv,
        target_class=target_class
    )

    # logs
    for cname, lst in cand_by_cls.items():
        if lst:
            ts_str = ", ".join([f"{t:.2f}s" for (t,_,_) in lst])
            print(f"[INFO] candidates[{cname}] = {len(lst)} at [{ts_str}]")
    if not events:
        print(f"[WARN] {name}: no FINAL events. Consider lowering --peak/--alpha or disable --use-priors.")
    else:
        per_cls={}
        for t,c,p in events: per_cls.setdefault(c,[]).append((t,p))
        for c, lst in per_cls.items():
            ts = ", ".join([f"{t:.2f}s" for (t,_p) in lst])
            print(f"[INFO] FINAL[{c}] -> {len(lst)} event(s) at [{ts}]")

    # export
    os.makedirs(args.outdir, exist_ok=True)
    stem = Path(ear_path).stem
    png_path = os.path.join(args.outdir, f"annotated_{stem}.png")
    csv_path = os.path.join(args.outdir, f"events_{stem}.csv")
    plot_with_shaded_events(ear, fs, events, png_path, title=name, shade_half_width_s=args.shade_half, show=args.show)
    save_csv(events, csv_path, name)
    print(f"[INFO] Saved PNG: {png_path}")
    print(f"[INFO] Saved CSV: {csv_path}")

if __name__ == "__main__":
    main()
