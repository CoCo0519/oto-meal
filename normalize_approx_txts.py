#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
normalize_approx_txts.py
将 *_approx.txt 规范化为 UTF-8 + 纯数值（6列，TSV）到 ./_normalized_data/
- 解决 Windows / GBK 解码报错
- 跳过 # 注释、时间+中文列名等非数值行
- 仅保留能转成 float 的前 6 列；不足 6 列跳过、>6 取前 6 列
- 保留“耳道/喉咙”两类文件（喉咙用于事件对齐，训练使用耳道）
"""

from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import sys

def try_read_lines(fp: Path):
    encs = ['utf-8', 'utf-8-sig', 'gb18030', 'cp936', 'latin1']
    for enc in encs:
        try:
            with open(fp, 'r', encoding=enc, errors='strict') as f:
                return f.read().splitlines(), enc
        except Exception:
            continue
    # 最后兜底：忽略错误
    try:
        with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read().splitlines(), 'utf-8(ignore)'
    except Exception:
        return None, None

def is_numeric6(parts):
    if len(parts) < 6:
        return False
    try:
        for i in range(6):
            float(parts[i].replace(',', '.'))
        return True
    except Exception:
        return False

def normalize_file(src: Path, dst: Path):
    lines, used_enc = try_read_lines(src)
    if lines is None:
        return 0, used_enc, "decode_fail"

    out = []
    for ln in lines:
        if not ln.strip():
            continue
        if ln.lstrip().startswith('#'):
            continue
        parts = [p for p in ln.strip().replace('\u3000', ' ').split() if p]
        if not is_numeric6(parts):
            continue
        try:
            vals = [str(float(parts[i].replace(',', '.'))) for i in range(6)]
            out.append('\t'.join(vals))
        except Exception:
            continue

    if not out:
        return 0, used_enc, "no_numeric_rows"

    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(out) + '\n')
    return len(out), used_enc, "ok"

def count_pairs(dir_path: Path) -> int:
    names = [f.name for f in dir_path.glob("*.txt")]
    ear = set(n for n in names if "耳道" in n)
    thr = set(n for n in names if "喉咙" in n)
    def keyify(n): 
        return n.replace("耳道", "###").replace("喉咙", "###")
    return len({keyify(n) for n in ear} & {keyify(n) for n in thr})

def main():
    ap = argparse.ArgumentParser(description="规范化 *_approx.txt 到 _normalized_data（UTF-8 + 数值6列）")
    ap.add_argument("--in-dir", type=str, required=True, help="源数据目录，如 ./denoised_hyx_data")
    ap.add_argument("--out-dir", type=str, default="./_normalized_data", help="输出目录（默认 ./_normalized_data）")
    ap.add_argument("--pattern", type=str, default="*_approx.txt", help="匹配的文件模式（默认 *_approx.txt）")
    args = ap.parse_args()

    src = Path(args.in_dir)
    if not src.exists() or not src.is_dir():
        print(f"❌ 输入目录不存在：{src.resolve()}")
        sys.exit(1)

    out = Path(args.out_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    files = list(src.rglob(args.pattern))
    print(f"[INFO] 在 {src.resolve()} 下匹配到 {len(files)} 个文件（pattern={args.pattern}）")
    if not files:
        print("⚠️ 未找到匹配文件，退出。")
        sys.exit(1)

    ok_files = 0
    total_rows = 0
    skipped = 0

    for fp in files:
        dst = out / fp.name
        n, enc, status = normalize_file(fp, dst)
        if status == "ok":
            ok_files += 1
            total_rows += n
            print(f"[OK]  {fp.name}  enc={enc}  ->  rows={n}")
        else:
            skipped += 1
            print(f"[SKIP] {fp.name}  enc={enc}  reason={status}")

    pairs = count_pairs(out)
    print(f"[INFO] 规范化完成：成功 {ok_files} / 跳过 {skipped}，总写入 {total_rows} 行，耳-喉可配对 {pairs} 对")
    print(f"[INFO] 规范化输出目录：{out.resolve()}")

if __name__ == "__main__":
    main()
