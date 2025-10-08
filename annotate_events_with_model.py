# -*- coding: utf-8 -*-
"""
annotate_events_with_model.py  (Conservative eventization to reduce false positives)

What’s new (vs. previous version):
- Robust eventization to avoid over-detection:
  * Adaptive per-class threshold: max(peak, median + alpha*MAD, q-quantile)
  * Connected-component on boolean mask to enforce MIN DURATION per event
  * Classification margin: prob(class) - max(other) >= margin
  * Temporal merge/NMS: merge events closer than --merge-within (keep higher prob)
  * Optional priors: --use-priors -> apply Top-K + min-spacing ONLY to target class inferred from filename;
    others are suppressed (or capped by --max-non-target)
- Window preprocessing matches training: resample each window to win_s * target_fs_win points.
- Rich [INFO] logs: thresholds used, candidate/filtered counts, final timestamps per class.

Examples:
  # Batch annotate all "耳道-*.txt" under a folder (more conservative defaults)
  python annotate_events_with_model.py --data-root ./denoised_hyx_data --use-priors --peak 0.65 --alpha 3.0 --q 0.9 --min-dur 0.8 --merge-within 1.0 --min-sep 8

  # Single file with slightly looser threshold
  python annotate_events_with_model.py --ear-file ./denoised_hyx_data/耳道-吞咽6次间隔10秒_approx.txt --use-priors --peak 0.6

Notes:
- We do NOT denoise again (your inputs are already denoised). We only detrend + z-score for model input.
"""

import os, re, glob, argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import scipy.signal as sg
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- Priors & label helpers ----------
PRIORS_COUNT_PAT = re.compile(r"(?:共)?(\d+)\s*次")
PRIORS_IV_PAT    = re.compile(r"间隔\s*(\d+)\s*秒")
PRIORS_DUR_PAT   = re.compile(r"持续\s*(\d+)\s*秒")

# Map filename keywords -> canonical class names
FNAME2CLASS = {
    "静息": "rest", "rest": "rest",
    "咀嚼": "chew", "chew": "chew",
    "咳嗽": "cough", "cough": "cough",
    "吞咽": "swallow", "喝水": "swallow", "swallow": "swallow", "drink": "swallow",
    "说话": "speak", "讲话": "speak", "speaking": "speak", "speak": "speak", "talk": "speak",
}

def infer_target_class_from_name(name: str, class_names):
    low = name.lower()
    # try CN/EN keywords
    for k, v in FNAME2CLASS.items():
        if k in low:
            # return only if in model classes
            if v in class_names:
                return v
    # fallback: first matching model class token in filename
    for cn in class_names:
        if cn in low:
            return cn
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
    x = sg.detrend(x, axis=0)
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
        x = self.conv(x)
        x = self.proj(x)
        x = x.transpose(1,2)
        x = self.enc(x)
        x = x.transpose(1,2)
        return self.cls(x)

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
        seg = resample_fixed_length(seg, int(round(win_s*target_fs_win)))  # consistency with training
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
    """Return list of (start_idx, end_idx_exclusive) for True runs."""
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
                              target_class=None, max_non_target=0):
    """
    Return: events = [(time_s, class_name, prob), ...] (sorted by time)
    Strategy:
      - Smooth prob per class.
      - Threshold = max(peak, median + alpha*MAD, q-quantile(track)).
      - Components (>=min_dur_s) -> take argmax as event.
      - Require classification margin.
      - Merge events closer than merge_within_s.
      - If use_priors and target_class & K_prior: keep only target_class, apply Top-K + spacing (≈Iv_prior or min_separation).
        Non-target classes limited by max_non_target.
    """
    events=[]
    if probs.shape[0]==0: return events
    C=probs.shape[1]

    # time resolution
    if len(times_s)>1:
        dt = float(np.median(np.diff(times_s)))
    else:
        dt = 0.25
    min_dur_pts = max(1, int(min_dur_s / max(dt,1e-6)))
    merge_pts   = max(1, int(merge_within_s / max(dt,1e-6)))
    min_sep_for_topk = Iv_prior if (use_priors and Iv_prior) else merge_within_s

    # smoothing kernel ~ 3 windows
    def smooth(y,k=3):
        if k<=1: return y
        ker = np.ones(min(k, len(y))) / max(1, min(k, len(y)))
        return np.convolve(y, ker, mode="same")

    # per-class candidates
    per_class_candidates = {}
    for ci in range(C):
        cname = class_names[ci] if ci < len(class_names) else f"class{ci}"
        if cname in ("rest","other"):  # ignore background classes
            continue
        track = smooth(probs[:,ci], k=3)
        # adaptive threshold
        thr_mad = np.median(track) + alpha * mad(track)
        thr_q   = np.quantile(track, q)
        thr = max(peak, thr_mad, thr_q)

        mask = track >= thr
        runs = connected_components(mask)
        cand=[]
        for (s,e) in runs:
            if (e - s) < min_dur_pts:
                continue
            j = s + int(np.argmax(track[s:e]))
            p = float(track[j])
            # classification margin: class prob vs best other
            margin_val = float(p - np.max(probs[j, np.arange(C)!=ci])) if C>1 else p
            if margin_val < margin:
                continue
            cand.append((float(times_s[j]), cname, p, j))
        per_class_candidates[cname] = cand

    # merge near-duplicate events across classes
    # first flatten, then sort by time, then greedy merge
    all_cands=[]
    for cname, lst in per_class_candidates.items():
        all_cands += [(t,c,p,idx) for (t,c,p,idx) in lst]
    all_cands.sort(key=lambda x: x[0])

    merged=[]
    for ev in all_cands:
        if not merged:
            merged.append(ev); continue
        (t,c,p,idx) = ev
        (t0,c0,p0,idx0) = merged[-1]
        if abs(t - t0) <= merge_within_s:
            # keep higher prob; if tie, prefer target_class if exists
            if p > p0 or (abs(p - p0) < 1e-6 and target_class and c == target_class and c0 != target_class):
                merged[-1] = ev
        else:
            merged.append(ev)

    # apply priors if requested: keep ONLY target_class and Top-K with spacing
    final=[]
    if use_priors and target_class and K_prior:
        # filter for target class
        tgt = [(t,c,p) for (t,c,p,idx) in merged if c == target_class]
        # top-k with spacing ~ Iv_prior (or min_sep_for_topk)
        if len(tgt) > 0:
            times = np.array([t for (t,_,_) in tgt], dtype=float)
            scores= np.array([p for (_,_,p) in tgt], dtype=float)
            # greedy
            order = np.argsort(scores)[::-1]
            keep=[]
            for k in order:
                if any(abs(times[k]-times[j]) < (Iv_prior*0.6 if Iv_prior else min_sep_for_topk) for j in keep):
                    continue
                keep.append(k)
                if len(keep) >= K_prior:
                    break
            # if still >K (due to no spacing), truncate
            keep = sorted(keep, key=lambda x: times[x])
            final = [(times[i], target_class, scores[i]) for i in keep[:K_prior]]
        else:
            final = []
        # optionally allow a few non-target (e.g., 0 by default)
        if max_non_target>0:
            nt = [(t,c,p) for (t,c,p,idx) in merged if c != target_class]
            nt.sort(key=lambda x: -x[2])
            final += nt[:max_non_target]
        # sort by time
        final.sort(key=lambda x: x[0])
    else:
        # no priors: just keep merged
        final = [(t,c,p) for (t,c,p,idx) in merged]

    return final, per_class_candidates, dict(
        dt=dt, min_dur_pts=min_dur_pts, thr_params=dict(peak=peak, alpha=alpha, q=q),
        merge_pts=merge_pts
    )

# ---------- Plot & CSV ----------
def plot_and_save(ear: np.ndarray, fs:int, events, out_png: str, title:str=""):
    t = np.arange(len(ear))/fs
    plt.figure(figsize=(12,4))
    y = ear[:,1]  # IR PPG as representative
    plt.plot(t, y, linewidth=1)
    ymax = np.max(y) if len(y)>0 else 1.0
    for (ts, cname, prob) in events:
        plt.axvline(ts, linestyle="--", linewidth=1)
        plt.text(ts, ymax, f"{cname}@{ts:.2f}s", rotation=90, fontsize=8,
                 va="bottom", ha="center")
    plt.xlabel("Time (s)"); plt.ylabel("Amplitude")
    if title: plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def save_csv(events, out_csv: str, file_name:str):
    import csv
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w=csv.writer(f)
        w.writerow(["file","time_s","class","prob"])
        for ts, cname, prob in events:
            w.writerow([file_name, f"{ts:.4f}", cname, f"{prob:.4f}"])

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, default=None, help="目录下所有耳道*approx.txt")
    ap.add_argument("--ear-file", type=str, default=None, help="只处理单个耳道文件")
    ap.add_argument("--model", type=str, default="artifacts/ear_conv_transformer.pt")
    ap.add_argument("--fs", type=int, default=0, help="如为0则自动估计(基于文件名先验+长度)")
    ap.add_argument("--win", type=float, default=2.0)
    ap.add_argument("--hop", type=float, default=0.25)
    ap.add_argument("--target-fs-win", type=int, default=100, help="推理窗重采样到该采样率xwin秒")

    # Conservative eventization params (tune to reduce false positives)
    ap.add_argument("--peak", type=float, default=0.65, help="基础概率阈值（更高=更保守）")
    ap.add_argument("--alpha", type=float, default=3.0, help="自适应阈值的MAD系数")
    ap.add_argument("--q", type=float, default=0.90, help="分位数阈值 q")
    ap.add_argument("--min-dur", type=float, default=0.8, help="事件最小持续时间(秒)")
    ap.add_argument("--margin", type=float, default=0.20, help="分类置信边际 prob(class)-max(other)")
    ap.add_argument("--merge-within", type=float, default=1.0, help="近距事件合并(秒)")
    ap.add_argument("--min-sep", type=float, default=8.0, help="Top-K最小间隔(秒)，用于先验")

    # Priors
    ap.add_argument("--use-priors", action="store_true", help="按文件名先验(K/Iv)对目标类别做Top-K+间隔约束")
    ap.add_argument("--max-non-target", type=int, default=0, help="使用先验时，允许的非目标类别最大事件数(默认0=抑制)")

    ap.add_argument("--outdir", type=str, default="annotate_outputs")
    ap.add_argument("--allow-cpu", action="store_true")
    args = ap.parse_args()

    # Load model
    ckpt = torch.load(args.model, map_location="cpu")
    class_names = ckpt.get("class_names", [f"class{i}" for i in range(5)])
    num_classes = len(class_names)
    model = ConvTransformer(in_ch=6, num_classes=num_classes)
    model.load_state_dict(ckpt["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() and not args.allow_cpu else "cpu")
    model.to(device)
    print(f"[INFO] Model loaded. classes={class_names} device={device.type}")

    # Collect files
    files=[]
    if args.ear_file:
        files=[args.ear_file]
    elif args.data_root:
        files=sorted([p for p in glob.glob(os.path.join(args.data_root, "*approx.txt"))
                      if "耳道" in Path(p).name])
    else:
        raise SystemExit("[ERROR] Provide --ear-file or --data-root")

    os.makedirs(args.outdir, exist_ok=True)

    for fp in files:
        name = Path(fp).name
        ear = read_approx_txt(fp)
        K,Iv,Dur = parse_priors_from_name(name)
        # fs
        fs = args.fs if args.fs>0 else estimate_fs_from_length_and_priors(len(ear), K,Iv,Dur, fallback=50)
        # infer target class from filename
        # (if not in model classes, target_class=None)
        # 先验只作用于“目标类别”
        target_class = infer_target_class_from_name(name, class_names)

        print(f"[INFO] {name}: fs={fs}Hz priors(K={K},Iv={Iv},Dur={Dur}) target_class={target_class}")

        # Sliding probs
        times, probs = sliding_predict_proba(model, ear, fs, device,
                                             win_s=args.win, hop_s=args.hop, target_fs_win=args.target_fs_win)
        print(f"[INFO] {name}: produced {len(times)} windows.")

        # Conservative eventization
        events, cand_by_cls, dbg = conservative_eventization(
            times, probs, class_names,
            peak=args.peak, alpha=args.alpha, q=args.q,
            min_dur_s=args.min_dur, margin=args.margin,
            merge_within_s=args.merge_within,
            use_priors=args.use_priors, K_prior=K, Iv_prior=Iv,
            target_class=target_class, max_non_target=args.max_non_target
        )

        # INFO: thresholds & candidate stats
        for cname, lst in cand_by_cls.items():
            if not lst: continue
            ts_str = ", ".join([f"{t:.2f}s" for (t,_,_,__) in lst])
            print(f"[INFO] {name}: candidates[{cname}] = {len(lst)} at [{ts_str}]")
        if len(events)==0:
            print(f"[WARN] {name}: no events after conservative filtering. Try lowering --peak (e.g., 0.55) "
                  f"or --alpha (e.g., 2.5), or disable --use-priors.")
        else:
            # Print final per-class summary
            per_cls={}
            for t,c,p in events: per_cls.setdefault(c,[]).append((t,p))
            for c, lst in per_cls.items():
                ts = ", ".join([f"{t:.2f}s" for (t,_p) in lst])
                print(f"[INFO] {name}: FINAL[{c}] -> {len(lst)} event(s) at [{ts}]")

        # Export
        stem = Path(fp).stem
        out_csv = os.path.join(args.outdir, f"events_{stem}.csv")
        out_png = os.path.join(args.outdir, f"annotated_{stem}.png")
        # For plotting, show ALL final events
        save_csv(events, out_csv, name)
        plot_and_save(ear, fs, events, out_png, title=name)
        print(f"[INFO] Saved CSV: {out_csv}")
        print(f"[INFO] Saved PNG: {out_png}")

if __name__ == "__main__":
    main()
