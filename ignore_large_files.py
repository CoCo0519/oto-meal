#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ignore_large_files.py
功能：扫描目标目录，查找超过阈值（默认 100MB）的文件，将其相对路径追加写入 .gitignore（避免重复）。
Feature: Scan a target directory to find files exceeding a threshold (default 100MB) and append their relative
paths to .gitignore (avoid duplicates).

用法示例 | Usage examples:
    python ignore_large_files.py
    python ignore_large_files.py --root /path/to/repo
    python ignore_large_files.py --root . --threshold 100MB
    python ignore_large_files.py --threshold 1GiB
"""

from __future__ import annotations
import argparse
import os
import sys
import time
from pathlib import Path

# 默认跳过的目录名（可按需增删）
# Default directories to skip (adjust as needed)
SKIP_DIRS = {'.git', '.hg', '.svn', '.venv', 'venv', 'node_modules', '.idea', '.mypy_cache', '__pycache__'}

# 支持的单位及其换算
# Supported units and their multipliers
SIZE_UNITS = {
    'b': 1,
    'kb': 1000, 'kib': 1024,
    'mb': 1000**2, 'mib': 1024**2,
    'gb': 1000**3, 'gib': 1024**3,
    'tb': 1000**4, 'tib': 1024**4,
}

def parse_size(s: str) -> int:
    """
    解析尺寸字符串为字节数，支持如：100M, 100MB, 750MiB, 1G, 1GiB, 500MB 等。
    Parse human size string to bytes: 100M, 100MB, 750MiB, 1G, 1GiB, 500MB, etc.
    """
    s = s.strip().lower().replace(' ', '')
    # 允许纯数字，按字节
    # Allow pure digits as bytes
    if s.isdigit():
        return int(s)

    # 拆分数值与单位
    # Split number and unit
    num_part = ''
    unit_part = ''
    for ch in s:
        if (ch.isdigit() or ch == '.' or ch == '+'):
            num_part += ch
        else:
            unit_part += ch

    if not num_part:
        raise ValueError(f"Invalid size: {s}")

    # 规范化常见缩写：m/mb -> mb；g/gb -> gb；k/kb -> kb
    # Normalize shortcuts
    unit_part = unit_part.replace('ib', 'iB').lower()  # keep case-insensitive
    # map common forms
    unit_map = {
        'k': 'kb', 'kb': 'kb', 'kib': 'kib',
        'm': 'mb', 'mb': 'mb', 'mib': 'mib',
        'g': 'gb', 'gb': 'gb', 'gib': 'gib',
        't': 'tb', 'tb': 'tb', 'tib': 'tib',
        'b': 'b',
    }

    # 去掉末尾的 'b' 变体处理（例如“mb”、“mib”）
    # Keep unit as-is if in dictionary; else try to coerce
    unit = unit_part
    if unit not in unit_map:
        # 常见变体：'mib', 'mb', 'm'，或者包含大小写 'B'
        unit = unit.replace('bytes', 'b').replace('byte', 'b')
        unit = unit.rstrip('s')  # remove plural 's'
    unit = unit_map.get(unit, unit)

    if unit not in SIZE_UNITS:
        raise ValueError(f"Unsupported unit in size: {s} (parsed unit: {unit})")

    multiplier = SIZE_UNITS[unit]
    # 支持小数，但最终转为 int 字节数
    # Allow decimal number then cast to int bytes
    bytes_val = float(num_part) * multiplier
    return int(bytes_val)

def normalize_rel_path(path: Path, root: Path) -> str:
    """
    将绝对路径转换为相对于 root 的 POSIX 风格相对路径。
    Convert absolute path to POSIX-style relative path from root.
    """
    rel = path.relative_to(root).as_posix()
    # 统一前缀（不以 ./ 开头）
    # Ensure no leading "./"
    if rel.startswith('./'):
        rel = rel[2:]
    return rel

def load_existing_gitignore_lines(gitignore_path: Path) -> list[str]:
    """
    读取 .gitignore 的现有行（保留原始行），失败则返回空列表。
    Read existing .gitignore lines; on failure return empty list.
    """
    if not gitignore_path.exists():
        return []
    try:
        with gitignore_path.open('r', encoding='utf-8', errors='replace') as f:
            return [line.rstrip('\n') for line in f]
    except Exception:
        # 作为兜底，用系统默认编码读取
        # Fallback with default encoding
        with gitignore_path.open('r', errors='replace') as f:
            return [line.rstrip('\n') for line in f]

def already_listed(existing_lines: list[str], rel_posix: str) -> bool:
    """
    粗略判断该相对路径是否已存在于 .gitignore（精确匹配; 不解读通配符）。
    Roughly check if rel path string is already present (exact match; no glob logic).
    """
    candidates = {rel_posix, f"./{rel_posix}", f"/{rel_posix}"}
    existing_set = set(line.strip() for line in existing_lines)
    return not candidates.isdisjoint(existing_set)

def append_gitignore(gitignore_path: Path, new_paths: list[str]) -> None:
    """
    将新路径追加写入 .gitignore，并带注释块。
    Append new paths to .gitignore with a comment header.
    """
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    header = [
        "",
        f"# ---- Added by ignore_large_files.py at {ts} ----",
        "# The following files exceeded the size threshold and were auto-appended.",
    ]
    # 确保父目录存在（通常是项目根）
    # Ensure parent exists (usually repo root)
    gitignore_path.parent.mkdir(parents=True, exist_ok=True)
    with gitignore_path.open('a', encoding='utf-8') as f:
        for line in header:
            f.write(line + "\n")
        for p in new_paths:
            f.write(p + "\n")

def find_large_files(root: Path, threshold_bytes: int) -> list[Path]:
    """
    扫描 root 下所有文件，返回超过阈值的文件路径列表。
    Walk under root and return file paths whose size exceeds threshold.
    """
    large = []
    # 使用 os.walk 以获得更高遍历性能
    # Use os.walk for speed
    for dirpath, dirnames, filenames in os.walk(root):
        # 过滤跳过目录
        # Prune skip dirs
        base_names = set(dirnames)
        to_remove = [d for d in base_names if d in SKIP_DIRS]
        for d in to_remove:
            dirnames.remove(d)

        for name in filenames:
            p = Path(dirpath) / name
            try:
                # 跳过符号链接（可选）
                # Skip symlinks (optional)
                if p.is_symlink():
                    continue
                sz = p.stat().st_size
                if sz > threshold_bytes:
                    large.append(p)
            except FileNotFoundError:
                # 文件被并发删除或不可达，忽略
                # File disappeared during scan; ignore
                continue
            except PermissionError:
                # 无权限读取，忽略或可记录
                # Permission denied; ignore or log
                continue
    return large

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scan for files larger than a threshold (default 100MB) and append their paths to .gitignore."
    )
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Root directory to scan (default: current directory).",
    )
    parser.add_argument(
        "--threshold",
        type=str,
        default="100MB",
        help="Size threshold, e.g. 100MB, 1GiB, 750MiB, 104857600 (bytes). Default: 100MB",
    )
    parser.add_argument(
        "--gitignore",
        type=str,
        default=None,
        help="Path to .gitignore file (default: <root>/.gitignore).",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        print(f"[ERROR] Root path does not exist: {root}", file=sys.stderr)
        return 2

    try:
        threshold_bytes = parse_size(args.threshold)
    except Exception as e:
        print(f"[ERROR] Invalid threshold '{args.threshold}': {e}", file=sys.stderr)
        return 2

    gitignore_path = Path(args.gitignore).resolve() if args.gitignore else (root / ".gitignore")

    print(f"Scanning root: {root}")
    print(f"Threshold: {threshold_bytes} bytes")
    print(f".gitignore: {gitignore_path}")

    large_files = find_large_files(root, threshold_bytes)
    print(f"Total large files found (> threshold): {len(large_files)}")

    if not large_files:
        print("No files exceed the threshold. Nothing to update.")
        return 0

    # 读取已存在的 .gitignore 行
    # Read existing lines
    existing_lines = load_existing_gitignore_lines(gitignore_path)

    # 规范化为相对路径（POSIX 风格），去重 & 去已存在项
    # Normalize to POSIX relative paths, de-duplicate & remove already-listed
    new_rel_paths = []
    seen = set()
    for p in large_files:
        rel = normalize_rel_path(p, root)
        if rel in seen:
            continue
        seen.add(rel)
        if not already_listed(existing_lines, rel):
            new_rel_paths.append(rel)

    if not new_rel_paths:
        print("All large files are already listed in .gitignore (or matched by broader patterns).")
        return 0

    append_gitignore(gitignore_path, new_rel_paths)
    print("Appended paths to .gitignore:")
    for rp in new_rel_paths:
        print(f"  - {rp}")

    print("\nNOTE:")
    print("  If some of these files are already tracked by Git, .gitignore won't untrack them automatically.")
    print("  You may need to run: git rm --cached <file>  (and then commit).")
    return 0

if __name__ == "__main__":
    sys.exit(main())
