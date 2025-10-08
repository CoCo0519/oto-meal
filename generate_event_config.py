#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
generate_event_config.py
Scan all *_denoise.txt under --data-dir and create/update events_config.json.

Features
- Incremental: if --out JSON exists, reuse it and APPEND ONLY new *_denoise files (old entries untouched).
- Auto parse from filename (both English & Chinese hints):
  * label: chew/drink/cough/swallow/talk/other
  * count:  共6次 / 6次 / 6x / 六次 ...
  * interval: 间隔10秒 / 每隔5秒 / 每5秒(一次) / 2.5s / 2.5sec ...
  * per-event duration (optional): 每次1秒 / 持续2秒 / 1s each ...
  * TOTAL duration (optional): 共31秒 / 总时长31秒 / 31s total ...
- Starts/Total generation:
  * If interval Δ and total T -> starts=[Δ,2Δ,...,⌊T/Δ⌋·Δ] (default not including T itself), total=T
  * Else if count N and interval Δ -> starts=[Δ,2Δ,...,NΔ], total=N·Δ
  * Add --start-at-zero to start from 0: [0,Δ,2Δ,...]
- You can edit the JSON afterwards.

Examples
  python generate_event_config.py --data-dir ./denoised_hyx_data --out ./events_config.json
  python generate_event_config.py --data-dir ./denoised_hyx_data --out ./events_config.json --start-at-zero
"""

from __future__ import annotations
import argparse
from pathlib import Path
import json
import re
import sys
from typing import Optional, Tuple, Dict, Any, List

ALLOWED_LABELS = ["chew", "drink", "cough", "swallow", "talk", "other"]

# ---------- Chinese numerals (0-99) ----------
_CN_DIGITS = {'零':0,'〇':0,'一':1,'二':2,'两':2,'三':3,'四':4,'五':5,'六':6,'七':7,'八':8,'九':9}
def _cn_to_int_token(tok: str) -> Optional[int]:
    tok = tok.strip()
    if not tok:
        return None
    if re.fullmatch(r'\d+', tok):
        return int(tok)
    if tok in _CN_DIGITS:
        return _CN_DIGITS[tok]
    if '十' in tok:
        if tok == '十':
            return 10
        left, *right = tok.split('十')
        left = left or ''
        right = right[0] if right else ''
        tens = 1 if left == '' else _CN_DIGITS.get(left, None)
        if tens is None:
            return None
        ones = 0 if right == '' else _CN_DIGITS.get(right, None)
        if ones is None:
            return None
        return tens * 10 + ones
    return None

def _extract_first_number(text: str) -> Optional[float]:
    m = re.search(r'(\d+(?:\.\d+)?)', text)
    if m:
        return float(m.group(1))
    m2 = re.search(r'([零〇一二两三四五六七八九十]+)', text)
    if m2:
        v = _cn_to_int_token(m2.group(1))
        if v is not None:
            return float(v)
    return None

# ---------- label detection ----------
_LABEL_MAP: List[Tuple[str, str]] = [
    (r'(咀嚼|嚼|chew|eat)', 'chew'),
    (r'(喝水|饮水|喝|drink)', 'drink'),
    (r'(咳嗽|咳|cough)', 'cough'),
    (r'(吞咽|咽|swallow)', 'swallow'),
    (r'(说话|讲话|说|talk|speak)', 'talk'),
]
def guess_label_from_name(name: str, default_label: str) -> str:
    base = name.lower()
    for pat, lab in _LABEL_MAP:
        if re.search(pat, base, flags=re.IGNORECASE):
            return lab
    return default_label if default_label in ALLOWED_LABELS else "chew"

# ---------- parse filename (count/interval/duration/total) ----------
def parse_filename_for_events(basename: str) -> Dict[str, Any]:
    """
    Supports:
      count: 共N次 / N次 / (\d+)[xX] / 中文数字
      interval: 间隔X秒 / 每隔X秒 / 每X秒(一次) / X秒 / Xs / Xsec
      per-event duration: 每次Y秒 / 持续Y秒 / Ys each / Ysec each
      total duration: 共T秒(钟) / 总时长T秒 / 总T秒 / T s total
    """
    name = basename
    name = re.sub(r'_denoise\.txt$', '', name, flags=re.IGNORECASE)
    name = name.replace('_', ' ').replace('-', ' ').strip()

    # count
    count = None
    for pat in [r'共\s*([零〇一二两三四五六七八九十\d]+)\s*次',
                r'([零〇一二两三四五六七八九十\d]+)\s*次',
                r'(\d+)\s*[xX]']:
        m = re.search(pat, name)
        if m:
            count = _extract_first_number(m.group(1))
            break

    # interval (also matches "每5秒一次")
    interval = None
    for pat in [r'(?:间隔|每隔|每)\s*([\d\.]+)\s*(?:秒|s|sec)(?:\s*一次)?',
                r'([\d\.]+)\s*(?:秒|s|sec)\s*(?:间隔|一?次)?']:
        m = re.search(pat, name, flags=re.IGNORECASE)
        if m:
            interval = float(m.group(1)); break
    if interval is None:
        m = re.search(r'([零〇一二两三四五六七八九十]+)\s*秒', name)
        if m:
            v = _cn_to_int_token(m.group(1))
            if v is not None:
                interval = float(v)

    # per-event duration
    dur_each = None
    for pat in [r'(?:每次|持续)\s*([\d\.]+)\s*(?:秒|s|sec)',
                r'([\d\.]+)\s*(?:秒|s|sec)\s*(?:each)']:
        m = re.search(pat, name, flags=re.IGNORECASE)
        if m:
            dur_each = float(m.group(1)); break

    # total duration
    total_sec = None
    for pat in [r'共\s*([\d\.]+)\s*秒(?:钟)?',
                r'总(?:时长)?\s*([\d\.]+)\s*秒',
                r'([\d\.]+)\s*(?:s|sec)\s*(?:total)']:
        m = re.search(pat, name, flags=re.IGNORECASE)
        if m:
            total_sec = float(m.group(1)); break
    if total_sec is None:
        m = re.search(r'共\s*([零〇一二两三四五六七八九十]+)\s*秒', name)
        if m:
            v = _cn_to_int_token(m.group(1))
            if v is not None:
                total_sec = float(v)

    return {
        "count": int(count) if count is not None else None,
        "interval_sec": float(interval) if interval is not None else None,
        "event_duration_each_sec": float(dur_each) if dur_each is not None else None,
        "total_sec": float(total_sec) if total_sec is not None else None,
    }

# ---------- utils ----------
def count_rows_txt(fp: Path) -> int:
    try:
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            return sum(1 for ln in f if ln.strip())
    except Exception:
        return 0

def load_existing_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def main():
    ap = argparse.ArgumentParser(description="Create/append events_config.json with filename auto-parsing")
    ap.add_argument("--data-dir", type=str, required=True, help="Root folder containing *_denoise.txt")
    ap.add_argument("--pattern", type=str, default="*_denoise.txt", help="Glob pattern (default *_denoise.txt)")
    ap.add_argument("--out", type=str, default="./events_config.json", help="Output JSON path (append if exists)")
    ap.add_argument("--default-label", type=str, default="chew", help="Default label if not recognized")
    ap.add_argument("--default-duration", type=float, default=2.0, help="Default per-event duration (sec)")
    ap.add_argument("--start-at-interval", action="store_true", default=True,
                    help="Start events at Δ (starts=[Δ,2Δ,...]; default)")
    ap.add_argument("--start-at-zero", action="store_true",
                    help="Start events at 0 (starts=[0,Δ,2Δ,...])")
    args = ap.parse_args()

    start_from_zero = bool(args.start_at_zero)

    root = Path(args.data_dir)
    if not root.exists():
        print(f"❌ Data dir not found: {root.resolve()}")
        sys.exit(1)

    files = sorted(list(root.rglob(args.pattern)))
    if not files:
        print(f"⚠️ No files matched: {args.pattern}")
        sys.exit(1)

    out_path = Path(args.out)
    existing = load_existing_json(out_path)
    existing_files_list: List[Dict[str, Any]] = []
    existing_names: set[str] = set()
    existing_version = 2

    if existing and isinstance(existing.get("files"), list):
        for it in existing["files"]:
            if isinstance(it, dict) and "file" in it and isinstance(it["file"], str):
                existing_files_list.append(it)
                existing_names.add(it["file"])
        existing_version = int(existing.get("version", 2)) if str(existing.get("version","2")).isdigit() else 2
        print(f"ℹ️ Reusing existing JSON: {out_path.resolve()} (already {len(existing_files_list)} entries)")

    auto_ok, auto_partial, auto_fail = 0, 0, 0
    skipped_because_exists = 0
    new_items: List[Dict[str, Any]] = []

    for fp in files:
        base = fp.name
        if base in existing_names:
            skipped_because_exists += 1
            continue

        rows = count_rows_txt(fp)
        guessed_label = guess_label_from_name(base, args.default_label)
        info = parse_filename_for_events(base)
        cnt = info["count"]
        interval = info["interval_sec"]
        dur_each = info["event_duration_each_sec"] if info["event_duration_each_sec"] is not None else float(args.default_duration)
        total = info["total_sec"]

        starts: List[float] = []
        notes: List[str] = []

        # Priority: (interval + total) -> (count + interval) -> partial -> none
        if interval is not None and total is not None and interval > 0 and total > 0:
            if start_from_zero:
                t = 0.0
                while t < total - 1e-9:
                    starts.append(round(t, 6))
                    t += interval
            else:
                t = interval
                while t < total + 1e-9:
                    if t < total - 1e-9:
                        starts.append(round(t, 6))
                    t += interval
            notes.append("auto:interval+total"); auto_ok += 1

        elif cnt is not None and interval is not None and cnt > 0 and interval > 0:
            if start_from_zero:
                starts = [round(i * interval, 6) for i in range(0, int(cnt))]
                total = (cnt - 1) * interval if cnt > 0 else 0.0
            else:
                starts = [round(i * interval, 6) for i in range(1, int(cnt) + 1)]
                total = cnt * interval
            notes.append("auto:count+interval"); auto_ok += 1

        elif (cnt is not None) ^ (interval is not None):
            notes.append("auto:count_only" if cnt is not None else "auto:interval_only")
            auto_partial += 1
        else:
            notes.append("auto:none"); auto_fail += 1

        item = {
            "file": base,
            "label": guessed_label,
            "event_duration_sec": float(dur_each),
            "event_starts_sec": starts,
            "total_duration_sec": (None if total is None else float(total)),
            "rows_detected": rows,
            "autofill": True,
            "autofill_notes": notes
        }
        new_items.append(item)

    merged_files = existing_files_list + new_items
    old_allowed = existing.get("allowed_labels") if isinstance(existing.get("allowed_labels"), list) else []
    merged_allowed = list(dict.fromkeys([*(old_allowed or []), *ALLOWED_LABELS]))
    note = existing.get("note") or (
        "Auto/incremental generated. Rules:\n"
        "- If interval Δ & total T: starts=[Δ,2Δ,...,⌊T/Δ⌋·Δ] (by default not including T itself);\n"
        "- If count N & interval Δ: starts=[Δ,2Δ,...,NΔ];\n"
        "- Use --start-at-zero for [0,Δ,2Δ,...];\n"
        "- Always review manually. Hz = rows_detected / total_duration_sec."
    )

    out = {
        "version": max(existing_version, 2),
        "allowed_labels": merged_allowed,
        "note": note,
        "files": merged_files
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"✅ Wrote: {out_path.resolve()}")
    print(f"  existing entries: {len(existing_files_list)}")
    print(f"  appended entries: {len(new_items)} (ok {auto_ok}; partial {auto_partial}; none {auto_fail})")
    if skipped_because_exists:
        print(f"  skipped (already present): {skipped_because_exists}")

if __name__ == "__main__":
    main()
