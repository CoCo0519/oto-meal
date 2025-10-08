# approx_txt_viewer.py
# GUI to scan and plot "Approximation TXT" files exported from MATLAB.
# - English-only UI
# - Chinese label support (auto-pick a CJK font if available)
# - Reads TXT with:
#     * 1st line optional comment starting with '#'
#     * next header line(s) for column names (default 1 header line after the comment)
#     * numeric matrix (N x M), e.g., PPG/IMU columns
# - Time axis: use time column if provided (1-based index), else synthesize from Hz
# - Each file renders as six-panel figures (2x3). If M>6, splits into multiple tabs.
#
# Dependencies: pip install numpy pandas matplotlib
# Run: python approx_txt_viewer.py

from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib
from matplotlib import font_manager

# ------------------ Chinese font setup ------------------
def setup_cjk_font():
    candidates = [
        "Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
        "Source Han Sans SC", "PingFang SC", "WenQuanYi Zen Hei",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            matplotlib.rcParams["font.family"] = [name]
            break
    matplotlib.rcParams["axes.unicode_minus"] = False

setup_cjk_font()

# ------------------ Helpers ------------------
def split_header_line(text: str, delim_choice: str):
    text = text.strip()
    if not text:
        return []
    if delim_choice == "auto":
        for sep in ["\t", ";", ","]:
            parts = [s for s in text.split(sep) if s != ""]
            if len(parts) > 1:
                return parts
        return text.split()  # whitespace
    elif delim_choice.startswith("tab"):
        return text.split("\t")
    elif delim_choice == "space":
        return text.split()
    elif delim_choice.startswith("comma"):
        return text.split(",")
    elif delim_choice.startswith("semicolon"):
        return text.split(";")
    else:
        return text.split()

def autodetect_header_names(path: Path, header_lines: int, delim_choice: str):
    """
    Read up to 'header_lines' lines, skipping lines that start with '#',
    and return names from the *last non-comment* header line if possible.
    """
    names = []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            seen = 0
            last_non_comment = None
            while seen < header_lines:
                line = f.readline()
                if not line:
                    break
                if line.lstrip().startswith("#"):
                    # comment line, do not count as a header line towards names
                    continue
                last_non_comment = line.rstrip("\n")
                seen += 1
            if last_non_comment:
                names = split_header_line(last_non_comment, delim_choice)
    except Exception:
        names = []
    return names

def read_numeric_block(path: Path, header_lines: int, delim_choice: str):
    """
    Read numeric matrix from approx TXT.
    - Treat lines starting with '#' as comments.
    - Skip 'header_lines' (non-comment) lines before numeric block.
    """
    # Compute how many physical lines to skip: we will use 'comment' to ignore '#',
    # so header_lines here refers to non-comment lines. We'll pre-scan to count
    # physical lines to skip before numeric content.
    skiprows_physical = 0
    noncomment_seen = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.lstrip().startswith("#"):
                skiprows_physical += 1
                continue
            # non-comment line
            skiprows_physical += 1
            noncomment_seen += 1
            if noncomment_seen >= header_lines:
                break

    common_kwargs = dict(header=None, engine="python", comment="#")
    if delim_choice == "auto":
        try:
            # whitespace/tab first
            df = pd.read_csv(path, skiprows=skiprows_physical, delim_whitespace=True, **common_kwargs)
        except Exception:
            try:
                df = pd.read_csv(path, skiprows=skiprows_physical, sep=",", **common_kwargs)
            except Exception:
                df = pd.read_csv(path, skiprows=skiprows_physical, sep=";", **common_kwargs)
    elif delim_choice.startswith("tab"):
        df = pd.read_csv(path, skiprows=skiprows_physical, sep=r"\t", **common_kwargs)
    elif delim_choice == "space":
        df = pd.read_csv(path, skiprows=skiprows_physical, delim_whitespace=True, **common_kwargs)
    elif delim_choice.startswith("comma"):
        df = pd.read_csv(path, skiprows=skiprows_physical, sep=",", **common_kwargs)
    elif delim_choice.startswith("semicolon"):
        df = pd.read_csv(path, skiprows=skiprows_physical, sep=";", **common_kwargs)
    else:
        df = pd.read_csv(path, skiprows=skiprows_physical, **common_kwargs)

    # Coerce numerics; drop all-NaN columns; forward/back fill minimal gaps
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(axis=1, how="all").copy()
    df = df.fillna(method="ffill").fillna(method="bfill")
    return df

# ------------------ GUI ------------------
class ScrollableCheckList(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        self.scroll = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)

        self.inner.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scroll.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scroll.pack(side="right", fill="y")

        self.items = []  # list of (Path, tk.BooleanVar)

    def set_items(self, paths):
        for w in self.inner.winfo_children():
            w.destroy()
        self.items.clear()
        for p in paths:
            var = tk.BooleanVar(value=False)
            cb = ttk.Checkbutton(self.inner, text=str(p), variable=var)
            cb.pack(anchor="w")
            self.items.append((Path(p), var))

    def get_selected(self):
        return [p for (p, v) in self.items if v.get()]

    def select_all(self, val=True):
        for _, v in self.items:
            v.set(val)

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Approx TXT Viewer")
        self.geometry("1280x780")

        # ===== Top bar =====
        top = ttk.Frame(self, padding=6)
        top.pack(side="top", fill="x")

        self.root_dir = tk.StringVar(value=str(Path.cwd()))
        ttk.Label(top, text="Folder:").pack(side="left")
        ttk.Entry(top, textvariable=self.root_dir, width=60).pack(side="left", padx=4)
        ttk.Button(top, text="Browse…", command=self.choose_dir).pack(side="left", padx=4)

        self.recursive = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="Recursive", variable=self.recursive).pack(side="left", padx=8)

        self.only_approx = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="Only *_approx.txt", variable=self.only_approx).pack(side="left", padx=8)

        ttk.Button(top, text="Scan", command=self.scan).pack(side="left", padx=4)
        ttk.Button(top, text="All", command=lambda: self.filelist.select_all(True)).pack(side="left")
        ttk.Button(top, text="None", command=lambda: self.filelist.select_all(False)).pack(side="left", padx=2)

        # ===== Options =====
        opts = ttk.Frame(self, padding=6)
        opts.pack(side="top", fill="x")

        self.hz = tk.DoubleVar(value=100.0)
        self.time_col = tk.IntVar(value=0)      # 0 = none
        self.header_lines = tk.IntVar(value=2)  # default: 1 comment + 1 header line
        self.max_points = tk.IntVar(value=200000)

        ttk.Label(opts, text="Hz (0 = use time column):").pack(side="left")
        ttk.Entry(opts, textvariable=self.hz, width=8).pack(side="left", padx=4)

        ttk.Label(opts, text="Time column (1-based, 0 = none):").pack(side="left", padx=(12,2))
        ttk.Entry(opts, textvariable=self.time_col, width=6).pack(side="left", padx=4)

        ttk.Label(opts, text="Header lines (non-comment):").pack(side="left", padx=(12,2))
        ttk.Spinbox(opts, from_=0, to=50, textvariable=self.header_lines, width=6).pack(side="left", padx=4)

        ttk.Label(opts, text="Delimiter:").pack(side="left", padx=(12,2))
        self.delim = tk.StringVar(value="auto")
        ttk.Combobox(opts, textvariable=self.delim, width=12, state="readonly",
                     values=["auto","tab(\\t)","space","comma(,)","semicolon(;)"]).pack(side="left", padx=4)

        ttk.Label(opts, text="Max points per plot:").pack(side="left", padx=(12,2))
        ttk.Entry(opts, textvariable=self.max_points, width=10).pack(side="left", padx=4)

        ttk.Button(opts, text="Plot (Tabs)", command=self.plot_tabs).pack(side="left", padx=10)

        # ===== Split panes =====
        main = ttk.PanedWindow(self, orient="horizontal")
        main.pack(fill="both", expand=True)

        left = ttk.Frame(main, padding=6)
        right = ttk.Frame(main, padding=0)
        main.add(left, weight=1)
        main.add(right, weight=3)

        ttk.Label(left, text="TXT Files:").pack(anchor="w")
        self.filelist = ScrollableCheckList(left)
        self.filelist.pack(fill="both", expand=True, pady=6)

        self.tabs = ttk.Notebook(right)
        self.tabs.pack(fill="both", expand=True)

        self.scan()

    # --------- scanning ---------
    def choose_dir(self):
        d = filedialog.askdirectory(initialdir=self.root_dir.get())
        if d:
            self.root_dir.set(d)
            self.scan()

    def scan(self):
        root = Path(self.root_dir.get()).expanduser()
        if not root.exists():
            messagebox.showerror("Error", f"Folder not found: {root}")
            return
        pattern = "*_approx.txt" if self.only_approx.get() else "*.txt"
        paths = []
        if self.recursive.get():
            for p in root.rglob(pattern):
                if p.is_file():
                    paths.append(str(p))
        else:
            for p in root.glob(pattern):
                if p.is_file():
                    paths.append(str(p))
        paths.sort()
        self.filelist.set_items(paths)

    # --------- plotting ---------
    def plot_tabs(self):
        files = self.filelist.get_selected()
        if not files:
            messagebox.showinfo("Hint", "Please select at least one file.")
            return

        # clear old tabs
        for t in self.tabs.tabs():
            self.tabs.forget(t)

        hz = float(self.hz.get())
        time_col = int(self.time_col.get())
        header_lines = int(self.header_lines.get())
        delim = self.delim.get()
        max_pts = int(self.max_points.get())

        for path in files:
            try:
                names = autodetect_header_names(Path(path), header_lines, delim)
                df = read_numeric_block(Path(path), header_lines, delim)
                nrows, ncols = df.shape
                if nrows == 0 or ncols == 0:
                    raise ValueError("Empty data after parsing.")

                # time axis
                if time_col > 0 and time_col <= ncols:
                    t = df.iloc[:, time_col - 1].to_numpy(dtype=float)
                    y_cols = [i for i in range(ncols) if i != time_col - 1]
                    if np.nanmax(t) > 1e5:
                        t = t / 1000.0
                else:
                    if hz <= 0:
                        raise ValueError("Hz must be > 0 when no time column is used.")
                    t = np.arange(nrows, dtype=float) / hz
                    y_cols = list(range(ncols))

                # downsample
                step = max(1, int(np.ceil(len(t) / max_pts)))
                t_ds = t[::step]

                # group columns in 6s
                def chunks(lst, size):
                    for i in range(0, len(lst), size):
                        yield lst[i:i+size]

                groups = list(chunks(y_cols, 6))
                total_tabs = len(groups)

                for idx, group in enumerate(groups, start=1):
                    fig = Figure(figsize=(10, 7), dpi=100)
                    axes = []
                    rows = 2 if len(group) > 3 else 1
                    cols = min(3, len(group))
                    for r in range(rows):
                        for c in range(cols):
                            axes.append(fig.add_subplot(rows, cols, r*cols + c + 1))

                    for sp, col_idx in enumerate(group):
                        ax = axes[sp]
                        y = df.iloc[:, col_idx].to_numpy(dtype=float)[::step]
                        ax.plot(t_ds, y, linewidth=0.9)
                        title_txt = names[col_idx] if col_idx < len(names) else f"col{col_idx+1}"
                        ax.set_title(str(title_txt), fontsize=10)
                        ax.grid(True)
                        if sp < 3 and rows == 2:
                            ax.tick_params(labelbottom=False)

                    # turn off unused axes
                    for j in range(len(group), len(axes)):
                        axes[j].axis("off")

                    suffix = f" ({idx}/{total_tabs})" if total_tabs > 1 else ""
                    fig.suptitle(f"{Path(path).name}{suffix} | approx TXT", fontsize=12)

                    tab = ttk.Frame(self.tabs)
                    label = Path(path).name[-20:]
                    if total_tabs > 1:
                        label += f" ({idx}/{total_tabs})"
                    self.tabs.add(tab, text=label)
                    canvas = FigureCanvasTkAgg(fig, master=tab)
                    canvas.draw()
                    canvas.get_tk_widget().pack(fill="both", expand=True)
                    NavigationToolbar2Tk(canvas, tab)

            except Exception as e:
                messagebox.showerror("Error", f"Failed to plot: {path}\n\n{e}")
                continue

if __name__ == "__main__":
    App().mainloop()
