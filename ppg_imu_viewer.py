# ppg_imu_viewer_v2.py
# - English-only UI labels
# - Chinese label support in Matplotlib (auto-detect a CJK font)
# - Per-file six-panel plotting (one subplot per data column)

import os
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib
from matplotlib import font_manager

# ---------- Chinese font setup (labels/legends/titles can show Chinese) ----------
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
    # show minus sign correctly with CJK fonts
    matplotlib.rcParams["axes.unicode_minus"] = False

setup_cjk_font()


class ScrollableCheckList(ttk.Frame):
    """Scrollable checkbox list for many files."""
    def __init__(self, master):
        super().__init__(master)
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        self.scroll = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)

        self.inner.bind("<Configure>", lambda e: self.canvas.configure(
            scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scroll.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scroll.pack(side="right", fill="y")

        self.vars = []   # list[(path, tk.BooleanVar)]
        self.checks = []

    def set_items(self, paths):
        for w in self.inner.winfo_children():
            w.destroy()
        self.vars.clear()
        self.checks.clear()
        for p in paths:
            v = tk.BooleanVar(value=False)
            cb = ttk.Checkbutton(self.inner, text=str(p), variable=v)
            cb.pack(anchor="w")
            self.vars.append((Path(p), v))
            self.checks.append(cb)

    def get_selected(self):
        return [p for (p, v) in self.vars if v.get()]

    def select_all(self, value=True):
        for _, v in self.vars:
            v.set(value)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("TXT Signal Viewer")
        self.geometry("1280x760")

        # === Top controls ===
        top = ttk.Frame(self, padding=6)
        top.pack(side="top", fill="x")

        self.root_dir = tk.StringVar(value=str(Path.cwd()))
        ttk.Label(top, text="Folder:").pack(side="left")
        self.dir_entry = ttk.Entry(top, textvariable=self.root_dir, width=60)
        self.dir_entry.pack(side="left", padx=4)
        ttk.Button(top, text="Browse…", command=self.choose_dir).pack(side="left", padx=4)

        self.recursive = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="Recursive", variable=self.recursive).pack(side="left", padx=8)
        ttk.Button(top, text="Scan", command=self.scan_files).pack(side="left", padx=4)
        ttk.Button(top, text="All", command=lambda: self.filelist.select_all(True)).pack(side="left")
        ttk.Button(top, text="None", command=lambda: self.filelist.select_all(False)).pack(side="left", padx=2)

        # === Settings ===
        opts = ttk.Frame(self, padding=6)
        opts.pack(side="top", fill="x")

        self.hz = tk.DoubleVar(value=100.0)
        self.time_col = tk.IntVar(value=0)      # 0 = no time column (use Hz)
        self.header_lines = tk.IntVar(value=1)
        self.max_points = tk.IntVar(value=200000)  # cap points to avoid lag

        ttk.Label(opts, text="Hz (0 = use time column):").pack(side="left")
        ttk.Entry(opts, textvariable=self.hz, width=8).pack(side="left", padx=4)

        ttk.Label(opts, text="Time column (1-based, 0 = none):").pack(side="left", padx=(12, 2))
        ttk.Entry(opts, textvariable=self.time_col, width=6).pack(side="left", padx=4)

        ttk.Label(opts, text="Header lines:").pack(side="left", padx=(12, 2))
        ttk.Spinbox(opts, from_=0, to=50, textvariable=self.header_lines, width=6).pack(side="left", padx=4)

        ttk.Label(opts, text="Delimiter:").pack(side="left", padx=(12, 2))
        self.delim = tk.StringVar(value="auto")
        delim_cb = ttk.Combobox(opts, textvariable=self.delim, width=12, state="readonly",
                                values=["auto", "tab(\\t)", "space", "comma(,)", "semicolon(;)"])
        delim_cb.pack(side="left", padx=4)

        ttk.Label(opts, text="Max points per plot:").pack(side="left", padx=(12, 2))
        ttk.Entry(opts, textvariable=self.max_points, width=10).pack(side="left", padx=4)

        ttk.Button(opts, text="Plot (Tabs)", command=self.plot_selected_tabs)\
            .pack(side="left", padx=10)

        # === Splitter ===
        main = ttk.PanedWindow(self, orient="horizontal")
        main.pack(fill="both", expand=True)

        left = ttk.Frame(main, padding=6)
        right = ttk.Frame(main, padding=0)
        main.add(left, weight=1)
        main.add(right, weight=3)

        ttk.Label(left, text="TXT Files:").pack(anchor="w")
        self.filelist = ScrollableCheckList(left)
        self.filelist.pack(fill="both", expand=True, pady=6)

        # Tab area for plots
        self.tabs = ttk.Notebook(right)
        self.tabs.pack(fill="both", expand=True)

        # initial scan
        self.scan_files()

    # ----------------- UI helpers -----------------
    def choose_dir(self):
        d = filedialog.askdirectory(initialdir=self.root_dir.get())
        if d:
            self.root_dir.set(d)
            self.scan_files()

    def scan_files(self):
        root = Path(self.root_dir.get()).expanduser()
        if not root.exists():
            messagebox.showerror("Error", f"Folder not found: {root}")
            return
        paths = []
        if self.recursive.get():
            for p in root.rglob("*.txt"):
                if p.is_file():
                    paths.append(str(p))
        else:
            for p in root.glob("*.txt"):
                if p.is_file():
                    paths.append(str(p))
        paths.sort()
        self.filelist.set_items(paths)

    # ----------------- Data I/O -----------------
    def _split_header(self, text, delim):
        """Split a header line into tokens according to delimiter choice."""
        # for 'auto', try in order: tab, semicolon, comma, whitespace
        tried = []
        def _try(sep):
            parts = [s for s in text.strip().split(sep) if len(s) > 0]
            return parts if len(parts) > 1 else None

        if delim == "auto":
            for sep in ["\t", ";", ","]:
                p = _try(sep)
                if p is not None: return p
            # whitespace split (multi-space)
            return text.strip().split()
        elif delim.startswith("tab"):
            return text.strip().split("\t")
        elif delim == "space":
            return text.strip().split()
        elif delim.startswith("comma"):
            return text.strip().split(",")
        elif delim.startswith("semicolon"):
            return text.strip().split(";")
        else:
            return text.strip().split()

    def _read_txt(self, path, header_lines, delim):
        """
        Read txt into a numeric DataFrame and return (df, names).
        names: column labels (best effort from the last header line, may be shorter).
        """
        path = Path(path)

        # read header lines raw
        header_texts = []
        if header_lines > 0:
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    for _ in range(header_lines):
                        header_texts.append(f.readline().rstrip("\n"))
            except Exception:
                header_texts = []

        # read numeric block
        if delim == "auto":
            try:
                df = pd.read_csv(path, skiprows=header_lines, header=None,
                                 delim_whitespace=True, engine="python")
            except Exception:
                try:
                    df = pd.read_csv(path, skiprows=header_lines, header=None, sep=",", engine="python")
                except Exception:
                    df = pd.read_csv(path, skiprows=header_lines, header=None, sep=";", engine="python")
        elif delim.startswith("tab"):
            df = pd.read_csv(path, skiprows=header_lines, header=None, sep=r"\t", engine="python")
        elif delim == "space":
            df = pd.read_csv(path, skiprows=header_lines, header=None, delim_whitespace=True, engine="python")
        elif delim.startswith("comma"):
            df = pd.read_csv(path, skiprows=header_lines, header=None, sep=",", engine="python")
        elif delim.startswith("semicolon"):
            df = pd.read_csv(path, skiprows=header_lines, header=None, sep=";", engine="python")
        else:
            df = pd.read_csv(path, skiprows=header_lines, header=None, engine="python")

        # numeric coercion + cleanup
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(axis=1, how="all").copy()
        df = df.fillna(method="ffill").fillna(method="bfill")

        # best-effort names from last header line
        names = None
        if header_texts:
            tokens = self._split_header(header_texts[-1], delim)
            if tokens:
                names = tokens

        # pad or fallback
        ncols = df.shape[1]
        if not names or len(names) < ncols:
            # default names: col1..colN (preserve any parsed names in front)
            base = [f"col{i+1}" for i in range(ncols)]
            if names:
                for i, v in enumerate(names[:ncols]):
                    base[i] = v
            names = base[:ncols]
        else:
            names = names[:ncols]

        return df, names

    # ----------------- Plotting -----------------
    def plot_selected_tabs(self):
        files = self.filelist.get_selected()
        if not files:
            messagebox.showinfo("Hint", "Please select at least one file.")
            return

        # Clear old tabs
        for t in self.tabs.tabs():
            self.tabs.forget(t)

        hz = float(self.hz.get())
        time_col = int(self.time_col.get())
        header_lines = int(self.header_lines.get())
        delim = self.delim.get()
        max_pts = int(self.max_points.get())

        for path in files:
            try:
                df, names = self._read_txt(path, header_lines, delim)
                nrows, ncols = df.shape
                if ncols == 0 or nrows == 0:
                    raise ValueError("Empty data frame")

                # time axis
                if time_col > 0 and time_col <= ncols:
                    t = df.iloc[:, time_col - 1].to_numpy(dtype=float)
                    y_cols = [i for i in range(ncols) if i != time_col - 1]
                    if t.max() > 1e5:  # likely ms → s
                        t = t / 1000.0
                else:
                    if hz <= 0:
                        raise ValueError("Hz must be > 0 when no time column is used")
                    t = np.arange(nrows, dtype=float) / hz
                    y_cols = list(range(ncols))

                # decimate for performance
                step = max(1, int(np.ceil(len(t) / max_pts)))
                t = t[::step]

                # figure with 6 panels (or dynamic if not exactly 6)
                N = len(y_cols)
                # Prefer 2x3 grid (six-panel style); if >6, grow rows; if <6, leave blanks off.
                cols = 3
                rows = int(np.ceil(max(N, 6) / cols)) if N > 6 else 2
                if N <= 3:
                    rows = 1
                fig = Figure(figsize=(10, 7), dpi=100)

                axes = []
                for r in range(rows):
                    for c in range(cols):
                        axes.append(fig.add_subplot(rows, cols, r*cols + c + 1))
                # plot each series into its own subplot
                for idx, col_idx in enumerate(y_cols):
                    ax = axes[idx]
                    y = df.iloc[:, col_idx].to_numpy(dtype=float)[::step]
                    ax.plot(t, y, linewidth=0.9)
                    ax.set_title(str(names[col_idx]), fontsize=10)  # supports Chinese
                    ax.grid(True)
                    if idx // cols < rows - 1:
                        ax.tick_params(labelbottom=False)  # share-x look

                # turn off any unused axes
                for j in range(len(y_cols), len(axes)):
                    axes[j].axis("off")

                # overall title
                fig.suptitle(f"{Path(path).name}", fontsize=12)

                # embed into a tab
                tab = ttk.Frame(self.tabs)
                self.tabs.add(tab, text=Path(path).name[-20:])
                canvas = FigureCanvasTkAgg(fig, master=tab)
                canvas.draw()
                canvas.get_tk_widget().pack(fill="both", expand=True)
                NavigationToolbar2Tk(canvas, tab)

            except Exception as e:
                messagebox.showerror(
                    "Error",
                    f"Failed to plot: {path}\n\n{e}\n"
                    "Check delimiter/header/time column/Hz."
                )
                continue


if __name__ == "__main__":
    App().mainloop()
