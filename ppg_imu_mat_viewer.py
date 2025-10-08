# ppg_imu_mat_viewer.py
# GUI to scan .mat files, select via checkboxes, and plot each file as six-panel figures.
# - English-only UI
# - Chinese label support (auto-pick a CJK font if available)
# - Auto-detect data/time variables with overrides
# - Chunk columns in groups of 6 → multiple tabs per file when needed
#
# Dependencies:
#   pip install numpy scipy pandas matplotlib h5py
#
# Run:
#   python ppg_imu_mat_viewer.py

from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import scipy.io as sio
import h5py
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

# ------------------ MAT loaders ------------------
def _is_numeric_array(arr: np.ndarray) -> bool:
    if not isinstance(arr, np.ndarray):
        return False
    return arr.dtype.kind in ("i", "u", "f", "b")  # int, uint, float, bool

def loadmat_pre73(path: Path):
    """Load non-v7.3 MAT using scipy.io.loadmat; return dict name->np.ndarray."""
    data = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    out = {}
    for k, v in data.items():
        if k.startswith("__"):
            continue
        # structs / objects are not directly usable here
        if isinstance(v, np.ndarray) and _is_numeric_array(np.atleast_1d(v)):
            out[k] = np.array(v)
    return out

def loadmat_v73(path: Path):
    """Load v7.3 MAT (HDF5) using h5py; return dict name->np.ndarray (numeric only)."""
    out = {}
    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            try:
                arr = obj[()]
                # h5py sometimes returns bytes for strings; we ignore non-numerics
                if isinstance(arr, np.ndarray) and _is_numeric_array(np.atleast_1d(arr)):
                    # Ensure at least 1D np array
                    out[name.split('/')[-1]] = np.array(arr)
            except Exception:
                pass
    with h5py.File(path, "r") as f:
        f.visititems(visit)
    return out

def load_mat_any(path: Path):
    """Return dict of variable name -> ndarray (numeric only), handling both MAT flavors."""
    try:
        # h5py can cheaply tell if it's HDF5 (v7.3)
        if h5py.is_hdf5(path):
            vars_dict = loadmat_v73(path)
        else:
            vars_dict = loadmat_pre73(path)
    except Exception:
        # fallback attempts
        try:
            vars_dict = loadmat_pre73(path)
        except Exception:
            vars_dict = loadmat_v73(path)
    return vars_dict

# ------------------ Heuristics for choosing data/time ------------------
PREF_DATA_NAMES = ["data", "X", "Y", "y", "yden", "denoised", "ppg", "imu", "ppg_imu"]
PREF_TIME_NAMES = ["t", "time", "timestamp", "time_sec", "times"]

def pick_data_array(vars_dict: dict, user_name: str | None):
    """Pick a 2D numeric array as data (rows=samples, cols=channels)."""
    # user override
    if user_name:
        for k, v in vars_dict.items():
            if k.lower() == user_name.lower():
                A = np.array(v)
                if A.ndim == 1:
                    A = A.reshape(-1, 1)
                if A.ndim == 2:
                    # rows as samples
                    if A.shape[0] < A.shape[1]:
                        A = A.T
                    return k, A
                # Not 2D → reject and continue
        # if not found, fall through to auto

    # prefer named variables
    candidates = []
    for k, v in vars_dict.items():
        A = np.array(v)
        if A.ndim == 1:
            A = A.reshape(-1, 1)
        if A.ndim == 2 and A.size >= 8:
            r, c = A.shape
            if r < c:
                A = A.T
                r, c = A.shape
            score = r * c
            name_bias = -PREF_DATA_NAMES.index(k.lower()) if k.lower() in PREF_DATA_NAMES else 0
            candidates.append((score, name_bias, k, A))
    if not candidates:
        return None, None
    # sort by score desc, then name bias
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    _, _, kbest, Abest = candidates[0]
    return kbest, Abest

def pick_time_vector(vars_dict: dict, nsamples: int, user_name: str | None):
    """Return time vector or None; try user override, then prefer typical names, then length match."""
    def _shape_to_vec(a):
        arr = np.array(a)
        if arr.ndim == 1:
            return arr.astype(float)
        if arr.ndim == 2 and 1 in arr.shape:
            return arr.reshape(-1).astype(float)
        return None

    # user override
    if user_name:
        for k, v in vars_dict.items():
            if k.lower() == user_name.lower():
                t = _shape_to_vec(v)
                if t is not None and t.size == nsamples:
                    return k, t

    # by preferred names
    for pref in PREF_TIME_NAMES:
        for k, v in vars_dict.items():
            if k.lower() == pref:
                t = _shape_to_vec(v)
                if t is not None and t.size == nsamples:
                    return k, t

    # any vector matching length
    for k, v in vars_dict.items():
        t = _shape_to_vec(v)
        if t is not None and t.size == nsamples:
            return k, t

    return None, None

def pick_names_list(vars_dict: dict, ncols: int):
    """Try to find names/labels for columns."""
    CAND_NAMES = ["names", "labels", "ch_names", "columns", "varNames"]
    for cand in CAND_NAMES:
        for k, v in vars_dict.items():
            if k.lower() == cand.lower():
                arr = v
                try:
                    # convert MATLAB cellstr/object arrays to list of str
                    if isinstance(arr, np.ndarray):
                        lst = []
                        for x in arr.reshape(-1):
                            if isinstance(x, bytes):
                                lst.append(x.decode("utf-8", "ignore"))
                            else:
                                lst.append(str(x))
                        if len(lst) >= 1:
                            return (lst + [f"col{i+1}" for i in range(ncols)])[0:ncols]
                except Exception:
                    pass
    # fallback
    return [f"col{i+1}" for i in range(ncols)]

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

        self.vars = []

    def set_items(self, paths):
        for w in self.inner.winfo_children():
            w.destroy()
        self.vars.clear()
        for p in paths:
            v = tk.BooleanVar(value=False)
            cb = ttk.Checkbutton(self.inner, text=str(p), variable=v)
            cb.pack(anchor="w")
            self.vars.append((Path(p), v))

    def get_selected(self):
        return [p for (p, v) in self.vars if v.get()]

    def select_all(self, value=True):
        for _, v in self.vars:
            v.set(value)

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MAT Signal Viewer")
        self.geometry("1280x780")

        # Top controls
        top = ttk.Frame(self, padding=6)
        top.pack(side="top", fill="x")

        self.root_dir = tk.StringVar(value=str(Path.cwd()))
        ttk.Label(top, text="Folder:").pack(side="left")
        ttk.Entry(top, textvariable=self.root_dir, width=60).pack(side="left", padx=4)
        ttk.Button(top, text="Browse…", command=self.choose_dir).pack(side="left", padx=4)

        self.recursive = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="Recursive", variable=self.recursive).pack(side="left", padx=8)
        ttk.Button(top, text="Scan", command=self.scan_files).pack(side="left", padx=4)
        ttk.Button(top, text="All", command=lambda: self.filelist.select_all(True)).pack(side="left")
        ttk.Button(top, text="None", command=lambda: self.filelist.select_all(False)).pack(side="left", padx=2)

        # Settings
        opts = ttk.Frame(self, padding=6)
        opts.pack(side="top", fill="x")

        self.hz = tk.DoubleVar(value=100.0)      # used if no time vector
        self.time_var = tk.StringVar(value="")   # optional: name of time variable in MAT
        self.data_var = tk.StringVar(value="")   # optional: name of data matrix in MAT
        self.max_points = tk.IntVar(value=200000)

        ttk.Label(opts, text="Hz (used if no time var):").pack(side="left")
        ttk.Entry(opts, textvariable=self.hz, width=8).pack(side="left", padx=4)

        ttk.Label(opts, text="Time var name (blank=auto):").pack(side="left", padx=(12, 2))
        ttk.Entry(opts, textvariable=self.time_var, width=18).pack(side="left", padx=4)

        ttk.Label(opts, text="Data var name (blank=auto):").pack(side="left", padx=(12, 2))
        ttk.Entry(opts, textvariable=self.data_var, width=18).pack(side="left", padx=4)

        ttk.Label(opts, text="Max points per plot:").pack(side="left", padx=(12, 2))
        ttk.Entry(opts, textvariable=self.max_points, width=10).pack(side="left", padx=4)

        ttk.Button(opts, text="Plot (Tabs)", command=self.plot_selected_tabs).pack(side="left", padx=10)

        # Splitter
        main = ttk.PanedWindow(self, orient="horizontal")
        main.pack(fill="both", expand=True)

        left = ttk.Frame(main, padding=6)
        right = ttk.Frame(main, padding=0)
        main.add(left, weight=1)
        main.add(right, weight=3)

        ttk.Label(left, text="MAT Files:").pack(anchor="w")
        self.filelist = ScrollableCheckList(left)
        self.filelist.pack(fill="both", expand=True, pady=6)

        self.tabs = ttk.Notebook(right)
        self.tabs.pack(fill="both", expand=True)

        self.scan_files()

    # ------------- helpers -------------
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
            for p in root.rglob("*.mat"):
                if p.is_file():
                    paths.append(str(p))
        else:
            for p in root.glob("*.mat"):
                if p.is_file():
                    paths.append(str(p))
        paths.sort()
        self.filelist.set_items(paths)

    # ------------- plotting -------------
    def plot_selected_tabs(self):
        files = self.filelist.get_selected()
        if not files:
            messagebox.showinfo("Hint", "Please select at least one file.")
            return

        # clear old tabs
        for t in self.tabs.tabs():
            self.tabs.forget(t)

        hz = float(self.hz.get())
        time_name = self.time_var.get().strip()
        data_name = self.data_var.get().strip()
        max_pts = int(self.max_points.get())

        for path in files:
            try:
                vars_dict = load_mat_any(Path(path))
                if not vars_dict:
                    raise ValueError("No numeric arrays found in MAT.")

                kdata, A = pick_data_array(vars_dict, data_name or None)
                if A is None:
                    raise ValueError("Could not find a 2D numeric data array (try setting 'Data var name').")
                # rows as samples; ensure float
                A = np.asarray(A, dtype=float)
                nrows, ncols = A.shape

                # time vector
                kt, t = pick_time_vector(vars_dict, nrows, time_name or None)
                if t is None:
                    if hz <= 0:
                        raise ValueError("No time vector found; please set Hz > 0 or specify 'Time var name'.")
                    t = np.arange(nrows, dtype=float) / hz
                else:
                    # if it looks like milliseconds, convert to seconds
                    if np.nanmax(t) > 1e5:
                        t = t / 1000.0

                # names
                names = pick_names_list(vars_dict, ncols)

                # decimation
                step = max(1, int(np.ceil(len(t) / max_pts)))
                t_ds = t[::step]

                # chunk columns in groups of 6
                def chunks(lst, size):
                    for i in range(0, len(lst), size):
                        yield lst[i:i+size]

                col_indices = list(range(ncols))
                chunks6 = list(chunks(col_indices, 6))
                ntabs = len(chunks6)

                for tab_idx, group in enumerate(chunks6, start=1):
                    fig = Figure(figsize=(10, 7), dpi=100)
                    axes = []
                    rows = 2
                    cols = 3
                    for r in range(rows):
                        for c in range(cols):
                            axes.append(fig.add_subplot(rows, cols, r*cols + c + 1))

                    # plot each selected column into its own panel
                    for sp_idx, col_idx in enumerate(group):
                        ax = axes[sp_idx]
                        y_ds = A[:, col_idx][::step]
                        ax.plot(t_ds, y_ds, linewidth=0.9)
                        ax.set_title(str(names[col_idx]), fontsize=10)
                        ax.grid(True)
                        if sp_idx < 3:  # top row: hide x labels for a cleaner look
                            ax.tick_params(labelbottom=False)

                    # turn off any unused axes
                    for j in range(len(group), len(axes)):
                        axes[j].axis("off")

                    # figure title
                    suffix = f" [{tab_idx}/{ntabs}]" if ntabs > 1 else ""
                    fig.suptitle(f"{Path(path).name}{suffix}  |  data={kdata}  time={kt if kt else f'Hz={hz}'}", fontsize=12)

                    # embed tab
                    tab = ttk.Frame(self.tabs)
                    label = Path(path).name[-20:]
                    if ntabs > 1:
                        label += f" ({tab_idx}/{ntabs})"
                    self.tabs.add(tab, text=label)
                    canvas = FigureCanvasTkAgg(fig, master=tab)
                    canvas.draw()
                    canvas.get_tk_widget().pack(fill="both", expand=True)
                    NavigationToolbar2Tk(canvas, tab)

            except Exception as e:
                messagebox.showerror(
                    "Error",
                    f"Failed to plot: {path}\n\n{e}\n"
                    "Hints: set 'Data var name' / 'Time var name', or check Hz."
                )
                continue

if __name__ == "__main__":
    App().mainloop()
