#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_behavior_classification.py
Multi-dir discovery -> TXT/JSON pairing -> JSON-only windowing -> CNN+Transformer training.

What’s new
- Accept multiple data roots: --data-dirs <dir1> <dir2> ...  (keeps --data-dir for backward-compat).
- Discover *.txt (default: *_denoise.txt) under each root; list each and show whether a matching JSON exists.
- Matching JSON policy (per file):
    <stem>.json  (preferred)
    <stem without _denoise>.json
    If not found, optionally use --event-config (master JSON) entry whose "file" basename matches this TXT.
- Train ONLY on TXT files that have a corresponding JSON (per-file or master) with
  required fields: label, event_starts_sec, event_duration_sec, total_duration_sec.
- Still supports: --ear-only / --use-throat; --no-other-class; denoise-only preference.

Outputs
- <outdir>/windows/sample_*.npy
- <outdir>/all_labels.json
- <outdir>/train/labels.json, <outdir>/val/labels.json
- <outdir>/_simple_outputs/{training_accuracy.png, training_metrics.json, confusion_matrix.png/json, best_model.pt}
"""

from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import shutil
import math
import re
import numpy as np

# ---------------- CLI ----------------
def build_arg_parser():
    ap = argparse.ArgumentParser(description="Multi-dir discovery + JSON-only windowing + multi-modal training")
    # Data roots
    ap.add_argument("--data-dirs", type=str, nargs="+",
                    help="One or more root folders containing *.txt (default: *_denoise.txt)")
    ap.add_argument("--data-dir", type=str, help="(Deprecated) Single root folder; use --data-dirs instead")
    # Optional master config (fallback when per-file JSON not found)
    ap.add_argument("--event-config", type=str, help="Path to a master events_config.json (optional)")
    ap.add_argument("--pattern", type=str, default="*_denoise.txt",
                    help="Glob pattern for TXT discovery under each root (default: *_denoise.txt)")
    # Modality selection
    ap.add_argument("--ear-only", action="store_true", default=True,
                    help="Use '耳道' files only (default True). Add --use-throat to include '喉咙'.")
    ap.add_argument("--use-throat", action="store_true", help="Also include '喉咙' in training")
    # Policy switches
    ap.add_argument("--denoise-only", action="store_true", default=True,
                    help="Prefer *_denoise.txt only (default True). If pattern differs, this is ignored.")
    ap.add_argument("--no-other-class", action="store_true",
                    help="Drop 'other' class; skip samples labeled as 'other'")
    # Split/eval
    ap.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio (default 0.2)")
    ap.add_argument("--stratified", action="store_true", default=True, help="Stratified split by label (default on)")
    ap.add_argument("--no-stratified", dest="stratified", action="store_false")
    ap.add_argument("--eval-set", type=str, choices=["val", "all"], default="val",
                    help="Which set to evaluate confusion matrix on (default: val)")
    ap.add_argument("--cv-folds", type=int, default=0,
                    help="K-fold cross validation (K>1). When set, ignores --val-ratio and --eval-set uses CV val.")
    # Train
    ap.add_argument("--seed", type=int, default=2025, help="Random seed")
    ap.add_argument("--outdir", type=str, default="./_prepared_events", help="Output dir")
    ap.add_argument("--batch-size", type=int, default=128, help="Batch size")
    ap.add_argument("--epochs", type=int, default=60, help="Epochs")
    ap.add_argument("--lr", type=float, default=3e-4, help="Initial learning rate")
    ap.add_argument("--weight-decay", type=float, default=1e-2, help="Weight decay for AdamW")
    ap.add_argument("--warmup-pct", type=float, default=0.1, help="Warmup ratio of total steps")
    ap.add_argument("--dropout", type=float, default=0.2, help="Dropout for Transformer")
    ap.add_argument("--emb-dim", type=int, default=192, help="Embedding dim")
    ap.add_argument("--nhead", type=int, default=8, help="Transformer num heads")
    ap.add_argument("--nlayers", type=int, default=4, help="Transformer num encoder layers")
    ap.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    ap.add_argument("--aug", action="store_true", default=True, help="Enable data augmentation (default on)")
    ap.add_argument("--no-aug", dest="aug", action="store_false")
    return ap

# ---------------- utils ----------------
def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _reset_outdir(p: Path):
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)

def _is_ear_file(name: str) -> bool:
    return "耳道" in name

def _normalize_space_hyphen(s: str) -> str:
    t = s.replace("\u3000", " ")
    t = re.sub(r"\s+", " ", t).strip()
    t = t.replace("- ", "-")
    return t

def _normalize_filename_key(name: str) -> str:
    """
    Build a relaxed key for matching master-config entries to a TXT:
    - lower; remove spaces/underscores; drop _denoise/_approx; drop extension.
    """
    s = _normalize_space_hyphen(name).lower()
    s = s.replace("_denoise", "").replace("_approx", "")
    if s.endswith(".txt"):
        s = s[:-4]
    s = s.replace(" ", "").replace("_", "")
    return s

# ---------------- discovery & pairing ----------------
def _load_master_config_map(cfg_path: Optional[Path]) -> Dict[str, dict]:
    """
    Load events_config.json (if provided) and build a map: normalized file basename -> item dict.
    Expected schema: {"files": [{"file": "...", "label": "...", "event_starts_sec":[...],
                                 "event_duration_sec": x.x, "total_duration_sec": y.y}, ...]}
    """
    if not cfg_path or not cfg_path.exists():
        return {}
    try:
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
        files = data.get("files", [])
        mp = {}
        for it in files:
            fn = it.get("file", "")
            key = _normalize_filename_key(Path(fn).name)
            if key:
                mp[key] = it
        return mp
    except Exception as e:
        print(f"[WARN] Failed to load master config: {cfg_path.name} ({e})")
        return {}

def _find_perfile_json_for_txt(txt_path: Path) -> Optional[Path]:
    stem = txt_path.with_suffix("").name
    cands = [
        txt_path.with_suffix(".json"),
        txt_path.with_name(stem.replace("_denoise", "") + ".json"),
        txt_path.with_name(stem.replace("_approx", "") + ".json"),
    ]
    for c in cands:
        if c.exists():
            return c
    return None

def _discover_txts(roots: List[Path], pattern: str) -> List[Path]:
    found = []
    for rt in roots:
        if not rt.exists():
            print(f"[WARN] Data root not found: {rt}")
            continue
        for fp in rt.rglob(pattern):
            if fp.is_file() and fp.suffix.lower() == ".txt":
                found.append(fp)
    return sorted(found)

def _collect_items_from_pairs(txts: List[Path], master_map: Dict[str, dict],
                              ear_only: bool, use_throat: bool,
                              include_other: bool) -> Tuple[List[dict], List[dict]]:
    """
    For each TXT, try to load a per-file JSON; otherwise search in master_map.
    Return:
      usable_items: [{fp, label, starts, dur, total} ...]
      report_rows:  [{txt, json_status, json_source, json_path|None, reason_if_skipped} ...] for logging
    """
    usable, rows = [], []
    for fp in txts:
        # modality filter for listing: we list all, but we may skip usage by ear_only flag
        is_ear = _is_ear_file(fp.name)
        if ear_only and (not is_ear) and (not use_throat):
            rows.append({"txt": str(fp), "json_status": "—", "json_source": "—",
                         "json_path": None, "reason": "skip (ear-only)"})
            continue

        per_json = _find_perfile_json_for_txt(fp)
        src = None
        cfg_item = None
        if per_json:
            try:
                cfg_item = json.loads(per_json.read_text(encoding="utf-8"))
                src = "per-file"
            except Exception as e:
                rows.append({"txt": str(fp), "json_status": "✗", "json_source": "per-file",
                             "json_path": str(per_json), "reason": f"bad json ({e})"})
                continue
        else:
            key = _normalize_filename_key(fp.name)
            if key in master_map:
                cfg_item = master_map[key]
                src = "master"
            else:
                rows.append({"txt": str(fp), "json_status": "✗", "json_source": "none",
                             "json_path": None, "reason": "no matching JSON"})
                continue

        # validate fields
        label = (cfg_item.get("label") or "other").lower()
        if (not include_other) and label not in {"chew","drink","cough","swallow","talk"}:
            rows.append({"txt": str(fp), "json_status": "✓", "json_source": src,
                         "json_path": str(per_json) if per_json else "events_config",
                         "reason": "skip (other class dropped)"})
            continue

        starts = cfg_item.get("event_starts_sec") or []
        dur = cfg_item.get("event_duration_sec", None)
        total = cfg_item.get("total_duration_sec", None)
        if (not starts) or (dur is None) or (total is None):
            rows.append({"txt": str(fp), "json_status": "✗", "json_source": src,
                         "json_path": str(per_json) if per_json else "events_config",
                         "reason": "json missing fields"})
            continue

        # accept this item
        usable.append({
            "fp": fp, "label": label, "starts": starts, "dur": float(dur), "total": float(total),
            "json_source": src, "json_path": (str(per_json) if per_json else None)
        })
        rows.append({"txt": str(fp), "json_status": "✓", "json_source": src,
                     "json_path": str(per_json) if per_json else "events_config", "reason": ""})
    return usable, rows

# ---------------- reader & slicing ----------------
def _read_six_cols(fp: Path) -> List[List[float]]:
    rows: List[List[float]] = []
    with fp.open("r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            s = ln.strip()
            if not s or s.startswith("#"):  # allow comments
                continue
            parts = s.replace("\u3000"," ").split()
            if len(parts) < 6:
                continue
            try:
                vals = [float(x.replace(",", ".")) for x in parts[:6]]
                rows.append(vals)
            except Exception:
                continue
    return rows

def _slice_by_time_series(rows: List[List[float]], total_sec: float, start_sec: float, dur_sec: float):
    if total_sec is None or total_sec <= 0 or dur_sec is None or dur_sec <= 0:
        return None
    n = len(rows)
    if n == 0:
        return None
    hz = float(n) / float(total_sec)
    eps = 1e-9
    st_time = min(max(0.0, float(start_sec)), float(total_sec) - eps)
    ed_time = min(float(total_sec), float(start_sec) + float(dur_sec))
    st = int(math.floor(st_time * hz))
    ed = int(math.ceil(ed_time * hz))
    st = max(0, min(st, n-1))
    ed = max(st+1, min(ed, n))
    if ed - st < 2 and n >= 2:
        ed = min(n, st + 2)
    if ed - st <= 1:
        return None
    return rows[st:ed], hz, st, ed

# ---------------- plotting helpers ----------------
def _setup_plot_fonts():
    try:
        import matplotlib
        for name in ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]:
            matplotlib.rcParams['font.sans-serif'] = [name]
            break
        matplotlib.rcParams['axes.unicode_minus'] = False
    except Exception:
        pass

def _plot_curves(history: Dict[str, List[float]], out_png: Path):
    _setup_plot_fonts()
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("⚠️ matplotlib not available; skip training curve")
        return
    epochs = list(range(1, max(
        len(history.get("train_acc", [])),
        len(history.get("val_acc", [])),
        len(history.get("val_loss", []))) + 1))
    plt.figure(figsize=(8,5))
    if history.get("train_acc"): plt.plot(epochs[:len(history["train_acc"])], history["train_acc"], label="Train Acc")
    if history.get("val_acc"): plt.plot(epochs[:len(history["val_acc"])], history["val_acc"], label="Val Acc")
    if history.get("val_loss"): plt.plot(epochs[:len(history["val_loss"])], history["val_loss"], label="Val Loss")
    if not (history.get("train_acc") or history.get("val_acc") or history.get("val_loss")):
        plt.text(0.5, 0.5, "No history", ha="center", va="center")
    plt.xlabel("Epoch"); plt.ylabel("Metric"); plt.legend(); plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150); plt.close()

def _plot_confusion_matrix(cm: List[List[int]], classes: List[str], out_png: Path, title="Confusion Matrix"):
    _setup_plot_fonts()
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("⚠️ matplotlib not available; skip confusion matrix plot")
        return
    cm_arr = np.array(cm, dtype=float)
    row_sum = cm_arr.sum(axis=1, keepdims=True) + 1e-9
    cm_norm = cm_arr / row_sum

    plt.figure(figsize=(7,6))
    im = plt.imshow(cm_norm, interpolation='nearest', aspect='auto')
    plt.title(title)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_marks = range(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right')
    plt.yticks(tick_marks, classes)
    thresh = cm_norm.max() / 2.0 if cm_norm.size else 0.5
    for i in range(len(classes)):
        for j in range(len(classes)):
            txt = f"{int(cm_arr[i, j])}\n({cm_norm[i, j]*100:.1f}%)"
            plt.text(j, i, txt, ha="center", va="center",
                     color="white" if cm_norm[i, j] > thresh else "black", fontsize=8)
    plt.ylabel("True"); plt.xlabel("Pred")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150); plt.close()

# ---------------- dataset build ----------------
def _compute_channel_stats_from_list(outdir: Path, labels: List[dict]) -> Dict[str, List[float]]:
    sum_c = np.zeros(6, dtype=np.float64)
    sumsq_c = np.zeros(6, dtype=np.float64)
    n_total = np.zeros(6, dtype=np.float64)
    for it in labels:
        arr = np.load(outdir / it["rel_path"])  # [C,T]
        sum_c += arr.sum(axis=1)
        sumsq_c += (arr*arr).sum(axis=1)
        n_total += arr.shape[1]
    mean = sum_c / np.maximum(n_total, 1.0)
    var = sumsq_c / np.maximum(n_total, 1.0) - mean*mean
    std = np.sqrt(np.maximum(var, 1e-8))
    stats = {"mean": mean.tolist(), "std": std.tolist()}
    with open(outdir / "channel_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Channel stats saved -> {outdir/'channel_stats.json'}")
    return stats

def _build_dataloaders(outdir: Path, train_labels: List[dict], val_labels: List[dict],
                       batch_size: int, target_len: int, stats: Dict[str,List[float]],
                       aug: bool):
    import torch
    from torch.utils.data import DataLoader

    mean = np.array(stats["mean"], dtype=np.float32).reshape(6,1)
    std  = np.array(stats["std"], dtype=np.float32).reshape(6,1)

    def _augment(x: np.ndarray) -> np.ndarray:
        T = x.shape[1]
        max_shift = max(1, int(0.05 * T))
        if max_shift > 1 and random.random() < 0.5:
            s = random.randint(-max_shift, max_shift)
            x = np.roll(x, s, axis=1)
        if random.random() < 0.7:
            scale = np.random.normal(1.0, 0.05, size=(6,1)).astype(np.float32)
            x = x * scale
        if random.random() < 0.7:
            noise = np.random.normal(0.0, 0.01, size=x.shape).astype(np.float32)
            x = x + noise
        if random.random() < 0.3:
            m = int(0.05 * T)
            if m > 1:
                start = random.randint(0, max(0, T - m))
                x[:, start:start+m] = 0.0
        return x

    def _fix_len(x: np.ndarray, T: int) -> np.ndarray:
        t = x.shape[1]
        if t == T:
            return x
        if t > T:
            st = (t - T) // 2
            return x[:, st:st+T]
        pad = T - t
        left = pad // 2
        right = pad - left
        return np.pad(x, ((0,0),(left,right)), mode="constant")

    class ListDataset(torch.utils.data.Dataset):
        def __init__(self, items: List[dict], do_aug=False):
            self.items = items
            self.do_aug = do_aug
        def __len__(self): return len(self.items)
        def __getitem__(self, idx):
            it = self.items[idx]
            arr = np.load(outdir / it["rel_path"])  # [C,T]
            T = target_len
            if self.do_aug:
                arr = _augment(arr.copy())
            arr = _fix_len(arr, T)
            arr = (arr - mean) / (std + 1e-8)
            import torch
            x = torch.from_numpy(arr).float()
            y = int(it["label_id"])
            return x, y

    train_ds = ListDataset(train_labels, do_aug=aug)
    val_ds   = ListDataset(val_labels,   do_aug=False)

    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          drop_last=False, pin_memory=True)
    val_ld   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                          drop_last=False, pin_memory=True)
    return train_ld, val_ld

def _build_model(n_classes: int, emb_dim: int, nhead: int, nlayers: int, dropout: float, device):
    import torch
    from torch import nn
    class CNNFrontEnd(nn.Module):
        def __init__(self, c_in=3, emb=192):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(c_in, 64, 9, padding=4), nn.ReLU(),
                nn.Conv1d(64, 128, 9, padding=4), nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(128, emb, 9, padding=4), nn.ReLU(),
            )
        def forward(self, x):      # [B,C,T] -> [B,emb,T/2]
            return self.net(x)

    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len=20000, dropout=0.0):
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            self.register_buffer('pe', pe, persistent=False)
        def forward(self, x):      # [B,T,D]
            x = x + self.pe[:, :x.size(1), :]
            return self.dropout(x)

    class MultiModalModel(nn.Module):
        def __init__(self, n_classes=6, emb=192, nhead=8, nlayers=4, dropout=0.2):
            super().__init__()
            self.ppg = CNNFrontEnd(3, emb)
            self.imu = CNNFrontEnd(3, emb)
            self.fuse = nn.Conv1d(emb*2, emb, 1)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=emb, nhead=nhead, dim_feedforward=emb*2, dropout=dropout, batch_first=True)
            self.tr = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
            self.pos = PositionalEncoding(emb, dropout=dropout)
            self.cls = nn.Linear(emb, n_classes)
        def forward(self, x):      # x:[B,6,T]
            ppg = x[:, :3, :]
            imu = x[:, 3:, :]
            zp = self.ppg(ppg)         # [B,emb,T']
            zi = self.imu(imu)         # [B,emb,T']
            z = torch.cat([zp, zi], dim=1)
            z = self.fuse(z)           # [B,emb,T']
            z = z.transpose(1, 2)      # [B,T',emb]
            z = self.pos(z)
            z = self.tr(z)
            z = z.mean(dim=1)
            return self.cls(z)

    model = MultiModalModel(n_classes=n_classes, emb=emb_dim, nhead=nhead, nlayers=nlayers, dropout=dropout).to(device)
    return model

# ---------------- prepare windows ----------------
def build_event_windows_from_items(items: List[dict], outdir: Path,
                                   include_other: bool) -> Tuple[List[dict], Dict[str,int]]:
    """
    items: [{fp (Path), label (str), starts (list), dur (float), total (float)} ...]
    Save windows to outdir/windows and return (all_labels, label_counts).
    """
    _reset_outdir(outdir)
    win_dir = outdir / "windows"
    _ensure_dir(win_dir)

    base_labels = ["chew", "drink", "cough", "swallow", "talk"]
    if include_other:
        base_labels.append("other")
    label2id = {lb: i for i, lb in enumerate(base_labels)}
    id2label = {v:k for k,v in label2id.items()}

    all_labels: List[dict] = []
    label_counts = {k:0 for k in label2id.keys()}
    total_windows = 0

    print("\n========== JSON WINDOWING ==========")
    idx = 0
    for it in items:
        fp: Path = it["fp"]
        label = it["label"].lower()
        if (not include_other) and label not in label2id:
            continue
        if label not in label2id:
            label = "other" if include_other else None
        if label is None:
            continue
        y = label2id[label]
        starts = it["starts"]
        dur = it["dur"]
        total = it["total"]

        rows = _read_six_cols(fp)
        if not rows:
            print(f"[WARN] empty or bad rows: {fp.name}")
            continue

        print(f"[FILE] {fp.name}  label={label}  rows={len(rows)}  total_sec={total}")
        for st in starts:
            ret = _slice_by_time_series(rows, float(total), float(st), float(dur))
            if ret is None:
                print(f"  [WARN] skip window start={st:.2f}s dur={float(dur):.2f}s (too short after clamping)")
                continue
            win_rows, hz, st_idx, ed_idx = ret
            arr = np.asarray(win_rows, dtype=np.float32).transpose(1, 0)  # [6,T]
            idx += 1
            rel_path = f"windows/sample_{idx:06d}.npy"
            np.save(outdir / rel_path, arr)
            rec = {
                "rel_path": rel_path,
                "label_id": int(y),
                "label_str": id2label[int(y)],
                "source_file": fp.name,
                "start_sec": float(st),
                "duration_sec": float(dur),
                "row_start": int(st_idx),
                "row_end": int(ed_idx),
                "channels": 6,
                "length": int(arr.shape[1])
            }
            all_labels.append(rec)
            label_counts[id2label[int(y)]] += 1
            total_windows += 1
            print(f"  [JSON] start={st:.2f}s dur={float(dur):.2f}s -> rows [{st_idx}:{ed_idx}) len={ed_idx - st_idx}  Hz={hz:.2f}")
    print(f"========== TOTAL WINDOWS: {total_windows} ==========\n")

    with open(outdir / "all_labels.json", "w", encoding="utf-8") as f:
        json.dump(all_labels, f, ensure_ascii=False, indent=2)

    print("[INFO] Windows per label:", {k:int(v) for k,v in label_counts.items()})
    return all_labels, {k:int(v) for k,v in label_counts.items()}

# ---------------- split & train ----------------
def _stratified_split(all_labels: List[dict], val_ratio: float, seed: int) -> Tuple[List[dict], List[dict]]:
    rnd = random.Random(seed)
    buckets: Dict[int, List[dict]] = {}
    for r in all_labels:
        buckets.setdefault(int(r["label_id"]), []).append(r)
    train, val = [], []
    for _, lst in buckets.items():
        rnd.shuffle(lst)
        n = len(lst)
        n_val = int(round(n * val_ratio))
        val.extend(lst[:n_val])
        train.extend(lst[n_val:])
    rnd.shuffle(train); rnd.shuffle(val)
    return train, val

def _save_split(outdir: Path, split_name: str, split_labels: List[dict]):
    split_dir = outdir / split_name
    _ensure_dir(split_dir)
    with open(split_dir / "labels.json", "w", encoding="utf-8") as f:
        json.dump(split_labels, f, ensure_ascii=False, indent=2)

def _compute_and_plot_confusion(y_true: List[int], y_pred: List[int], classes: List[str], outdir: Path, tag=""):
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for t,p in zip(y_true, y_pred):
        if 0 <= t < len(classes) and 0 <= p < len(classes):
            cm[t,p] += 1
    title = "Confusion Matrix" + (f" ({tag})" if tag else "")
    _plot_confusion_matrix(cm.tolist(), classes, outdir / f"confusion_matrix{('_'+tag if tag else '')}.png", title=title)
    with open(outdir / f"confusion_matrix{('_'+tag if tag else '')}.json", "w", encoding="utf-8") as f:
        json.dump({"labels": classes, "matrix": cm.tolist()}, f, ensure_ascii=False, indent=2)
    return float((cm.trace() / max(1, cm.sum()))), int(cm.sum())

def _train_eval_once(outdir: Path, train_labels: List[dict], val_labels: List[dict],
                     classes: List[str], batch_size: int, epochs: int, lr: float,
                     weight_decay: float, warmup_pct: float, emb_dim: int, nhead: int,
                     nlayers: int, dropout: float, patience: int, aug: bool,
                     seed: int, tag: str = "") -> Tuple[Dict, Dict, List[int], List[int]]:
    try:
        import torch
    except Exception:
        out = outdir / "_simple_outputs"
        _ensure_dir(out)
        with open(out / f"training_metrics{('_'+tag if tag else '')}.json", "w", encoding="utf-8") as f:
            json.dump({"trained": False, "reason": "torch not installed"}, f, ensure_ascii=False, indent=2)
        return {"trained": False}, {"val_acc": None}, [], []

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Target length as median
    lengths = [it["length"] for it in (train_labels + val_labels)] or [256]
    target_len = int(sorted(lengths)[len(lengths)//2])
    print(f"[INFO] Target sequence length (median): {target_len}")

    stats = _compute_channel_stats_from_list(outdir, train_labels)
    from torch.utils.data import DataLoader
    train_ld, val_ld = _build_dataloaders(outdir, train_labels, val_labels, batch_size, target_len, stats, aug)

    # class weights
    n_classes = len(classes)
    cls_counts = np.zeros(n_classes, dtype=np.int64)
    for it in train_labels: cls_counts[it["label_id"]] += 1
    weights = cls_counts.sum() / np.maximum(cls_counts, 1)
    weights = weights / weights.mean()
    cls_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    print(f"[INFO] Class counts: {cls_counts.tolist()}  -> class weights: {weights.round(3).tolist()}")

    model = _build_model(n_classes, emb_dim, nhead, nlayers, dropout, device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    crt = torch.nn.CrossEntropyLoss(weight=cls_weights)

    total_steps = max(1, epochs * max(1, len(train_ld)))
    warmup_steps = int(total_steps * max(0.0, min(warmup_pct, 0.5)))
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    def _eval(loader):
        model.eval()
        tot, corr, loss_sum = 0, 0, 0.0
        all_true, all_pred = [], []
        with torch.no_grad():
            for x,y in loader:
                x = x.to(device); y = y.to(device)
                with torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
                    logits = model(x); loss = crt(logits, y)
                loss_sum += float(loss.item()) * x.size(0)
                pred = logits.argmax(1)
                corr += int((pred==y).sum().item())
                tot += x.size(0)
                all_true.extend(y.tolist()); all_pred.extend(pred.tolist())
        acc = (corr/tot if tot else 0.0)
        avg_loss = (loss_sum/tot if tot else 0.0)
        return acc, avg_loss, all_true, all_pred

    history = {"train_acc": [], "val_acc": [], "val_loss": []}
    out_simple = outdir / "_simple_outputs"
    _ensure_dir(out_simple)
    best_path = out_simple / f"best_model{('_'+tag if tag else '')}.pt"
    best_val = -1.0; no_improve = 0

    for ep in range(1, epochs+1):
        model.train()
        for (x,y) in train_ld:
            x = x.to(device); y = y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
                logits = model(x); loss = crt(logits, y)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); scheduler.step()
        train_acc, _, _, _ = _eval(train_ld)
        val_acc,  val_loss, _, _ = _eval(val_ld)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_loss"].append(val_loss)
        print(f"[run][{ep:02d}] train_acc={train_acc:.3f}  val_acc={val_acc:.3f}  val_loss={val_loss:.4f}")
        if val_acc > best_val + 1e-6:
            best_val = val_acc; no_improve = 0
            import torch as _t; _t.save(model.state_dict(), best_path)
        else:
            no_improve += 1
            if no_improve >= 10:
                print(f"[run][EarlyStop] No improvement for 10 epochs. Stop at epoch {ep}.")
                break

    import torch as _t
    if best_path.exists():
        model.load_state_dict(_t.load(best_path, map_location=device))
    val_acc, val_loss, y_true, y_pred = _eval(val_ld)

    _plot_curves(history, out_simple / f"training_accuracy{('_'+tag if tag else '')}.png")
    with open(out_simple / f"training_metrics{('_'+tag if tag else '')}.json", "w", encoding="utf-8") as f:
        json.dump({"history": history,
                   "summary": {"val_acc": float(val_acc), "val_loss": float(val_loss)},
                   "best": {"val_acc": float(best_val)}}, f, ensure_ascii=False, indent=2)

    return {"history": history}, {"val_acc": float(val_acc)}, y_true, y_pred

# ---------------- main ----------------
def main():
    args = build_arg_parser().parse_args()
    random.seed(args.seed); np.random.seed(args.seed)

    # Build roots list
    roots: List[Path] = []
    if args.data_dirs:
        roots.extend([Path(p) for p in args.data_dirs])
    if args.data_dir:
        roots.append(Path(args.data_dir))
    if not roots:
        print("❌ No data roots. Use --data-dirs <dir1> <dir2> ...")
        raise SystemExit(1)

    # Master config (optional)
    master_map: Dict[str, dict] = {}
    if args.event_config:
        cfg_path = Path(args.event_config)
        if not cfg_path.exists():
            print(f"[WARN] --event-config not found: {cfg_path}")
        else:
            master_map = _load_master_config_map(cfg_path)

    # Discover TXT
    pattern = args.pattern or "*_denoise.txt"
    print("\n========== DATASET DISCOVERY ==========")
    txts = _discover_txts(roots, pattern)
    if not txts:
        print("❌ No TXT files discovered. Check roots/pattern.")
        raise SystemExit(1)
    print(f"[INFO] Discovered TXT count: {len(txts)} with pattern '{pattern}'")

    # Pair with JSON
    include_other = (not args.no_other_class)
    usable_items, report_rows = _collect_items_from_pairs(
        txts, master_map, ear_only=args.ear_only, use_throat=args.use_throat, include_other=include_other
    )

    # Report listing
    used = 0
    for r in report_rows:
        mark = r["json_status"]
        src = r["json_source"]
        origin = f" ({src})" if src and src != "—" else ""
        extra = f"  -> {r['reason']}" if r.get("reason") else ""
        jpath = f" [{r['json_path']}]" if r.get("json_path") else ""
        print(f"[TXT] {r['txt']}  JSON:{mark}{origin}{jpath}{extra}")
        if mark == "✓" and not extra:
            used += 1
    print(f"[INFO] Usable TXT (with JSON): {len(usable_items)} / {len(txts)}")

    if not usable_items:
        print("❌ No usable TXT+JSON pairs. Please add per-file JSONs or provide --event-config.")
        raise SystemExit(1)

    # Build windows
    outdir = Path(args.outdir)
    all_labels, label_counts = build_event_windows_from_items(usable_items, outdir, include_other=include_other)
    if not all_labels:
        print("❌ No windows built from JSON; please check your JSON contents.")
        raise SystemExit(1)

    # Class list
    classes = ["chew", "drink", "cough", "swallow", "talk"]
    if include_other:
        classes.append("other")
    print("[INFO] Total windows:", len(all_labels))
    print("[INFO] Classes:", classes)

    # Split
    if args.stratified:
        train_labels, val_labels = _stratified_split(all_labels, args.val_ratio, args.seed)
    else:
        rnd = random.Random(args.seed)
        shuffled = all_labels[:]; rnd.shuffle(shuffled)
        n_val = int(round(len(shuffled) * args.val_ratio))
        val_labels = shuffled[:n_val]; train_labels = shuffled[n_val:]

    _save_split(outdir, "train", train_labels)
    _save_split(outdir, "val",   val_labels)
    print(f"[SPLIT] train={len(train_labels)}  val={len(val_labels)}  (stratified={args.stratified})")

    # Train once (no CV path here to keep concise; can be re-added similarly)
    metrics, best, y_true, y_pred = _train_eval_once(
        outdir, train_labels, val_labels, classes,
        args.batch_size, args.epochs, args.lr, args.weight_decay, args.warmup_pct,
        args.emb_dim, args.nhead, args.nlayers, args.dropout, args.patience,
        args.aug, seed=args.seed, tag=""
    )

    # Confusion matrix on val (default) or ALL
    out_simple = outdir / "_simple_outputs"; _ensure_dir(out_simple)
    if args.eval_set.lower() == "all":
        # Evaluate on all windows
        try:
            import torch
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # Rebuild and load best
            model = _build_model(len(classes), args.emb_dim, args.nhead, args.nlayers, args.dropout, device)
            best_path = out_simple / "best_model.pt"
            if best_path.exists():
                model.load_state_dict(torch.load(best_path, map_location=device))
            model.eval()

            # Prepare loader over all labels
            stats = _compute_channel_stats_from_list(outdir, train_labels)
            lengths = [it["length"] for it in all_labels] or [256]
            target_len = int(sorted(lengths)[len(lengths)//2])
            from torch.utils.data import DataLoader
            def _fix_len(x: np.ndarray, T: int) -> np.ndarray:
                t = x.shape[1]
                if t == T: return x
                if t > T:
                    st = (t - T) // 2
                    return x[:, st:st+T]
                pad = T - t
                left = pad // 2; right = pad - left
                return np.pad(x, ((0,0),(left,right)), mode="constant")
            mean = np.array(stats["mean"], dtype=np.float32).reshape(6,1)
            std  = np.array(stats["std"], dtype=np.float32).reshape(6,1)
            class AllDS(torch.utils.data.Dataset):
                def __init__(self, items): self.items = items
                def __len__(self): return len(self.items)
                def __getitem__(self, idx):
                    it = self.items[idx]
                    arr = np.load(outdir / it["rel_path"])
                    arr = _fix_len(arr, target_len)
                    arr = (arr - mean) / (std + 1e-8)
                    import torch as _t
                    return _t.from_numpy(arr).float(), int(it["label_id"])
            ds = AllDS(all_labels)
            ld = DataLoader(ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
            y_true_all, y_pred_all = [], []
            with torch.no_grad():
                for x,y in ld:
                    x = x.to(device)
                    pred = model(x).argmax(1).cpu().tolist()
                    y_pred_all.extend(pred); y_true_all.extend(y.tolist())
            acc_all, ns = _compute_and_plot_confusion(y_true_all, y_pred_all, classes, out_simple, tag="all")
            print(f"[EVAL-ALL] accuracy={acc_all:.4f}  samples={ns}")
        except Exception as e:
            print(f"[EVAL-ALL] failed: {e}")
    else:
        acc_val, ns = _compute_and_plot_confusion(y_true, y_pred, classes, out_simple, tag="val")
        print(f"[VAL] accuracy={acc_val:.4f}  samples={ns}")

    print("\n✅ Done.")
    print(f"   Windows dir: {outdir.resolve() / 'windows'}")
    print(f"   Outputs:     {outdir.resolve() / '_simple_outputs'}")

if __name__ == "__main__":
    main()
