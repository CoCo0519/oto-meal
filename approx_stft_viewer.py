# approx_stft_viewer_v13.py
# Core STFT changed to Matlab-like 'spectrogram' semantics.
# Buttons English-only; labels/titles support CJK.
# Deps: pip install numpy pandas matplotlib scipy

from pathlib import Path
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import pandas as pd
from scipy.signal import spectrogram, get_window, butter, filtfilt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib import font_manager
import matplotlib
import matplotlib.pyplot as plt

# ---------------- Font: CJK support ----------------
def setup_cjk_font():
    candidates = ["Microsoft YaHei","SimHei","Noto Sans CJK SC","Source Han Sans SC","PingFang SC","WenQuanYi Zen Hei"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            matplotlib.rcParams["font.family"] = [name]
            break
    matplotlib.rcParams["axes.unicode_minus"] = False
setup_cjk_font()

# ---------------- IO helpers ----------------
def _split_header_line(s: str, delim: str):
    s = s.strip()
    if not s: return []
    if delim == "auto":
        for sp in ["\t",";",","]:
            parts = [x for x in s.split(sp) if x!=""]
            if len(parts) > 1: return parts
        return s.split()
    if delim.startswith("tab"): return s.split("\t")
    if delim == "space": return s.split()
    if delim.startswith("comma"): return s.split(",")
    if delim.startswith("semicolon"): return s.split(";")
    return s.split()

def read_table_body(path: Path, header_lines: int, delim: str):
    # count physical skip rows, respecting leading '#'
    skip = 0; seen = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.lstrip().startswith("#"):
                skip += 1; continue
            skip += 1; seen += 1
            if seen >= header_lines: break
    kw = dict(header=None, engine="python", comment="#")
    if delim == "auto":
        try: df = pd.read_csv(path, skiprows=skip, delim_whitespace=True, **kw)
        except Exception:
            try: df = pd.read_csv(path, skiprows=skip, sep=",", **kw)
            except Exception: df = pd.read_csv(path, skiprows=skip, sep=";", **kw)
    elif delim.startswith("tab"): df = pd.read_csv(path, skiprows=skip, sep=r"\t", **kw)
    elif delim == "space":       df = pd.read_csv(path, skiprows=skip, delim_whitespace=True, **kw)
    elif delim.startswith("comma"): df = pd.read_csv(path, skiprows=skip, sep=",", **kw)
    elif delim.startswith("semicolon"): df = pd.read_csv(path, skiprows=skip, sep=";", **kw)
    else: df = pd.read_csv(path, skiprows=skip, **kw)
    for c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(axis=1, how="all").fillna(method="ffill").fillna(method="bfill")
    return df

def header_names(path: Path, header_lines: int, delim: str):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            seen=0; last=None
            while seen < header_lines:
                ln=f.readline()
                if not ln: break
                if ln.lstrip().startswith("#"): continue
                last=ln.rstrip("\n"); seen+=1
        return _split_header_line(last or "", delim)
    except Exception:
        return []

# ---------------- Signal helpers ----------------
def estimate_fs(t_raw: np.ndarray, time_unit: str):
    t_raw = np.asarray(t_raw, dtype=float).ravel()
    if time_unit == "auto":
        dt = np.diff(t_raw); dt = dt[np.isfinite(dt)]
        if dt.size == 0: return np.nan
        med = float(np.median(np.abs(dt)))
        if med >= 5e6: scale = 1e9
        elif med >= 5e3: scale = 1e6
        elif med >= 5:   scale = 1e3
        else:            scale = 1.0
        t = t_raw/scale
    else:
        scale = {"s":1.0,"ms":1e3,"us":1e6,"ns":1e9}[time_unit]
        t = t_raw/scale
    dt = np.diff(t); dt = dt[(dt>0) & np.isfinite(dt)]
    if dt.size == 0: return np.nan
    return float(1.0/np.median(dt))

def butter_zero(y, fs, hp, lp, order=4):
    y = np.asarray(y, dtype=float); nyq = 0.5*fs
    b=a=None
    if hp>0 and lp>0:
        lo=max(1e-6,hp/nyq); hi=min(0.999,lp/nyq)
        if lo<hi: 
            b,a = butter(order, [lo,hi], btype="band")
    elif hp>0: b,a = butter(order, max(1e-6,hp/nyq), btype="high")
    elif lp>0: b,a = butter(order, min(0.999,lp/nyq), btype="low")
    if b is not None:
        try: return filtfilt(b,a,y,axis=0,method="pad")
        except Exception: return y
    return y

from scipy.signal import butter, filtfilt  # after def

def next_pow2(n):
    n = int(max(1,n)); p=1
    while p<n: p <<= 1
    return p

def choose_clim_db(spec_db, dr=70.0, vmax_pct=99.8):
    vals = spec_db[np.isfinite(spec_db)]
    if vals.size < 16:
        return None, None
    vmax = np.nanpercentile(vals, vmax_pct)
    vmin = vmax - dr
    return vmin, vmax

# ---------------- UI list ----------------
class ScrollList(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        self.scroll = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner  = ttk.Frame(self.canvas)
        self.inner.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0,0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scroll.set)
        self.canvas.pack(side="left", fill="both", expand=True); self.scroll.pack(side="right", fill="y")
        self.items=[]
    def set_items(self, paths):
        for w in self.inner.winfo_children(): w.destroy()
        self.items.clear()
        for p in paths:
            v = tk.BooleanVar(value=False)
            ttk.Checkbutton(self.inner, text=str(p), variable=v).pack(anchor="w")
            self.items.append((Path(p), v))
    def get_selected(self): return [p for p,v in self.items if v.get()]
    def select_all(self, val=True):
        for _,v in self.items: v.set(val)

# ---------------- Main App ----------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Approx TXT STFT Viewer (v13, Matlab-like spectrogram core)")
        self.geometry("1680x980")

        # Top bar
        top = ttk.Frame(self, padding=6); top.pack(side="top", fill="x")
        self.root_dir = tk.StringVar(value=str(Path.cwd()))
        ttk.Label(top, text="Folder:").pack(side="left")
        ttk.Entry(top, textvariable=self.root_dir, width=60).pack(side="left", padx=4)
        ttk.Button(top, text="Browse…", command=self.choose_dir).pack(side="left", padx=4)
        self.recursive = tk.BooleanVar(value=True); ttk.Checkbutton(top, text="Recursive", variable=self.recursive).pack(side="left", padx=8)
        self.only_approx = tk.BooleanVar(value=True); ttk.Checkbutton(top, text="Only *_approx.txt", variable=self.only_approx).pack(side="left", padx=8)
        ttk.Button(top, text="Scan", command=self.scan).pack(side="left", padx=4)
        ttk.Button(top, text="All", command=lambda: self.filelist.select_all(True)).pack(side="left")
        ttk.Button(top, text="None", command=lambda: self.filelist.select_all(False)).pack(side="left", padx=2)

        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=8)
        ttk.Button(top, text="Load Config", command=self.load_cfg).pack(side="left", padx=2)
        ttk.Button(top, text="Save Config", command=self.save_cfg).pack(side="left", padx=2)
        ttk.Button(top, text="Apply Ref Sharp", command=self.apply_ref_sharp).pack(side="left", padx=8)

        # Parse/time
        opt = ttk.Frame(self, padding=6); opt.pack(side="top", fill="x")
        self.hz = tk.DoubleVar(value=100.0)
        self.time_col = tk.IntVar(value=0)
        self.header_lines = tk.IntVar(value=2)
        self.delim = tk.StringVar(value="auto")
        self.time_unit = tk.StringVar(value="auto")  # auto/s/ms/us/ns
        ttk.Label(opt, text="Hz (0=use time col):").pack(side="left")
        ttk.Entry(opt, textvariable=self.hz, width=8).pack(side="left", padx=4)
        ttk.Label(opt, text="Time col (1-based, 0=none):").pack(side="left", padx=(12,2))
        ttk.Entry(opt, textvariable=self.time_col, width=6).pack(side="left", padx=4)
        ttk.Label(opt, text="Header lines:").pack(side="left", padx=(12,2))
        ttk.Spinbox(opt, from_=0, to=50, textvariable=self.header_lines, width=6).pack(side="left", padx=4)
        ttk.Label(opt, text="Delimiter:").pack(side="left", padx=(12,2))
        ttk.Combobox(opt, textvariable=self.delim, width=12, state="readonly",
                     values=["auto","tab(\\t)","space","comma(,)","semicolon(;)"]).pack(side="left", padx=4)
        ttk.Label(opt, text="Time unit:").pack(side="left", padx=(12,2))
        ttk.Combobox(opt, textvariable=self.time_unit, width=8, state="readonly",
                     values=["auto","s","ms","us","ns"]).pack(side="left", padx=4)

        # Preprocess
        pre = ttk.Frame(self, padding=6); pre.pack(side="top", fill="x")
        self.remove_dc = tk.BooleanVar(value=True)
        self.hp = tk.DoubleVar(value=0.04)   # mild HP reduces 0 Hz leakage
        self.lp = tk.DoubleVar(value=0.0)    # off by default
        ttk.Checkbutton(pre, text="Remove DC", variable=self.remove_dc).pack(side="left")
        ttk.Label(pre, text="HP (Hz):").pack(side="left", padx=(12,2)); ttk.Entry(pre, textvariable=self.hp, width=8).pack(side="left")
        ttk.Label(pre, text="LP (Hz):").pack(side="left", padx=(12,2)); ttk.Entry(pre, textvariable=self.lp, width=8).pack(side="left")

        # STFT core (Matlab-like)
        core = ttk.Frame(self, padding=6); core.pack(side="top", fill="x")
        self.window = tk.StringVar(value="hamming")  # Matlab default is Hamming when integer
        self.nperseg = tk.IntVar(value=256)
        self.noverlap = tk.IntVar(value=224)  # 87.5%
        self.nfft = tk.IntVar(value=2048)
        self.mode = tk.StringVar(value="magnitude")  # magnitude or psd
        self.db_scale = tk.BooleanVar(value=True)
        ttk.Label(core, text="Window:").pack(side="left")
        ttk.Combobox(core, textvariable=self.window, width=12, state="readonly",
                     values=["hamming","hann","blackman","boxcar"]).pack(side="left", padx=4)
        ttk.Label(core, text="nperseg:").pack(side="left", padx=(12,2)); ttk.Entry(core, textvariable=self.nperseg, width=8).pack(side="left")
        ttk.Label(core, text="noverlap:").pack(side="left", padx=(12,2)); ttk.Entry(core, textvariable=self.noverlap, width=8).pack(side="left")
        ttk.Label(core, text="nfft:").pack(side="left", padx=(12,2)); ttk.Entry(core, textvariable=self.nfft, width=8).pack(side="left")
        ttk.Label(core, text="Mode:").pack(side="left", padx=(12,2))
        ttk.Combobox(core, textvariable=self.mode, width=10, state="readonly",
                     values=["magnitude","psd"]).pack(side="left", padx=4)
        ttk.Checkbutton(core, text="dB scale", variable=self.db_scale).pack(side="left", padx=12)

        # Display
        dsp = ttk.Frame(self, padding=6); dsp.pack(side="top", fill="x")
        self.fmin = tk.DoubleVar(value=0.0)
        self.fmax = tk.DoubleVar(value=6.0)
        self.render = tk.StringVar(value="imshow-bicubic")
        self.cmap = tk.StringVar(value="turbo")
        self.dr_db = tk.DoubleVar(value=70.0)
        self.vmax_pct = tk.DoubleVar(value=99.8)
        ttk.Label(dsp, text="Fmin:").pack(side="left"); ttk.Entry(dsp, textvariable=self.fmin, width=7).pack(side="left")
        ttk.Label(dsp, text="Fmax:").pack(side="left", padx=(10,2)); ttk.Entry(dsp, textvariable=self.fmax, width=7).pack(side="left")
        ttk.Label(dsp, text="Render:").pack(side="left", padx=(12,2))
        ttk.Combobox(dsp, textvariable=self.render, width=18, state="readonly",
                     values=["imshow-bicubic","imshow-bilinear","imshow-nearest","pcolormesh-gouraud","pcolormesh-nearest"]).pack(side="left", padx=2)
        ttk.Label(dsp, text="Colormap").pack(side="left", padx=(12,2))
        ttk.Combobox(dsp, textvariable=self.cmap, width=12, state="readonly",
                     values=["turbo","viridis","plasma","magma","inferno","cividis","jet"]).pack(side="left", padx=2)
        ttk.Label(dsp, text="dB range").pack(side="left", padx=(12,2)); ttk.Entry(dsp, textvariable=self.dr_db, width=7).pack(side="left")
        ttk.Label(dsp, text="vmax pct").pack(side="left", padx=(12,2)); ttk.Entry(dsp, textvariable=self.vmax_pct, width=7).pack(side="left")
        ttk.Button(dsp, text="Plot Spectrograms", command=self.plot_tabs).pack(side="left", padx=12)

        # Split main
        main = ttk.PanedWindow(self, orient="horizontal"); main.pack(fill="both", expand=True)
        left = ttk.Frame(main, padding=6); right = ttk.Frame(main, padding=0)
        main.add(left, weight=1); main.add(right, weight=3)
        ttk.Label(left, text="TXT Files:").pack(anchor="w")
        self.filelist = ScrollList(left); self.filelist.pack(fill="both", expand=True, pady=6)
        self.tabs = ttk.Notebook(right); self.tabs.pack(fill="both", expand=True)
        self.scan()

    # ---- scan & cfg ----
    def choose_dir(self):
        d = filedialog.askdirectory(initialdir=self.root_dir.get())
        if d: self.root_dir.set(d); self.scan()
    def scan(self):
        root = Path(self.root_dir.get())
        patt = "*_approx.txt" if self.only_approx.get() else "*.txt"
        if not root.exists(): messagebox.showerror("Error", f"Folder not found: {root}"); return
        paths = [str(p) for p in (root.rglob(patt) if self.recursive.get() else root.glob(patt)) if p.is_file()]
        paths.sort(); self.filelist.set_items(paths)

    def cfg_dict(self):
        return dict(
            root_dir=self.root_dir.get(), recursive=bool(self.recursive.get()), only_approx=bool(self.only_approx.get()),
            hz=float(self.hz.get()), time_col=int(self.time_col.get()), header_lines=int(self.header_lines.get()),
            delim=self.delim.get(), time_unit=self.time_unit.get(),
            remove_dc=bool(self.remove_dc.get()), hp=float(self.hp.get()), lp=float(self.lp.get()),
            window=self.window.get(), nperseg=int(self.nperseg.get()), noverlap=int(self.noverlap.get()),
            nfft=int(self.nfft.get()), mode=self.mode.get(), db_scale=bool(self.db_scale.get()),
            fmin=float(self.fmin.get()), fmax=float(self.fmax.get()),
            render=self.render.get(), cmap=self.cmap.get(), dr_db=float(self.dr_db.get()), vmax_pct=float(self.vmax_pct.get())
        )
    def load_cfg(self):
        p = filedialog.askopenfilename(title="Load Config", filetypes=[("JSON","*.json"),("All","*.*")])
        if not p: return
        try:
            cfg = json.load(open(p,"r",encoding="utf-8"))
            for k,v in self.cfg_dict().items():
                if k in cfg:
                    getattr(self, k).set(cfg[k]) if isinstance(getattr(self,k), tk.Variable) else None
            messagebox.showinfo("Config","Loaded.")
        except Exception as e:
            messagebox.showerror("Error", str(e))
    def save_cfg(self):
        p = filedialog.asksaveasfilename(title="Save Config", defaultextension=".json",
                                         filetypes=[("JSON","*.json"),("All","*.*")], initialfile="stft_config.json")
        if not p: return
        try:
            json.dump(self.cfg_dict(), open(p,"w",encoding="utf-8"), ensure_ascii=False, indent=2)
            messagebox.showinfo("Config","Saved.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # ---- Ref Sharp preset (focused on 0–6 Hz, ~0.6 s window) ----
    def apply_ref_sharp(self):
        files = self.filelist.get_selected()
        if files:
            df = read_table_body(files[0], int(self.header_lines.get()), self.delim.get())
            fs = self.infer_fs(df, df.shape[1])
        else:
            fs = float(self.hz.get()) if float(self.hz.get())>0 else 100.0
        nper = max(128, int(next_pow2(int(round(fs*0.6)))))  # ~0.6 s window
        nper = min(nper, 4096)
        self.window.set("hamming"); self.nperseg.set(nper); self.noverlap.set(int(round(nper*0.875)))
        self.nfft.set(max(nper, min(65536, next_pow2(nper*8))))
        self.remove_dc.set(True); self.hp.set(0.04); self.lp.set(0.0)
        self.mode.set("magnitude"); self.db_scale.set(True)
        self.fmin.set(0.0); self.fmax.set(6.0)
        self.render.set("imshow-bicubic"); self.cmap.set("turbo")
        self.dr_db.set(70.0); self.vmax_pct.set(99.8)
        messagebox.showinfo("Preset",
            f"Ref Sharp: fs≈{fs:.2f} Hz, nperseg={nper} (~{nper/fs:.2f}s), overlap≈87.5%, nfft={self.nfft.get()}")

    def infer_fs(self, df, ncols):
        tcol = int(self.time_col.get()); hz = float(self.hz.get())
        if tcol>0 and tcol<=ncols:
            fs = estimate_fs(df.iloc[:, tcol-1].to_numpy(dtype=float), self.time_unit.get())
            if np.isfinite(fs) and fs>0: return fs
        return hz if hz>0 else 100.0

    # ---- Matlab-like spectrogram core ----
    def compute_spectrogram_like_matlab(self, y, fs, window_name, nperseg, noverlap, nfft, mode):
        """
        Returns f, t, Spec
        - mode='magnitude' -> |S| (linear amplitude)
        - mode='psd'       -> PSD density (linear)
        """
        win = get_window(window_name, nperseg, fftbins=True)
        noverlap = int(np.clip(noverlap, 0, nperseg-1))
        if mode == "psd":
            f, t, Pxx = spectrogram(
                y, fs=fs, window=win, nperseg=nperseg, noverlap=noverlap,
                nfft=max(nperseg, nfft), detrend=False, return_onesided=True,
                scaling="density", mode="psd"
            )
            return f, t, Pxx
        else:
            # complex STFT then take magnitude -> matches 's' then abs(s)
            f, t, S = spectrogram(
                y, fs=fs, window=win, nperseg=nperseg, noverlap=noverlap,
                nfft=max(nperseg, nfft), detrend=False, return_onesided=True,
                scaling="spectrum", mode="complex"
            )
            return f, t, np.abs(S)

    # ---- Plotting ----
    def plot_tabs(self):
        files = self.filelist.get_selected()
        if not files:
            messagebox.showinfo("Hint","Select files first."); return
        for t in self.tabs.tabs(): self.tabs.forget(t)

        hz=float(self.hz.get()); tcol=int(self.time_col.get()); header=int(self.header_lines.get()); delim=self.delim.get()
        window=self.window.get(); nperseg=int(self.nperseg.get()); noverlap=int(self.noverlap.get()); nfft=int(self.nfft.get())
        mode=self.mode.get(); use_db=bool(self.db_scale.get())
        remove_dc=bool(self.remove_dc.get()); hp=float(self.hp.get()); lp=float(self.lp.get())
        fmin=float(self.fmin.get()); fmax=float(self.fmax.get())
        render=self.render.get(); cmap=self.cmap.get(); dr=float(self.dr_db.get()); vmax_pct=float(self.vmax_pct.get())
        tunit=self.time_unit.get()

        for path in files:
            try:
                names = header_names(path, header, delim)
                df = read_table_body(path, header, delim)
                nrows, ncols = df.shape
                if nrows==0 or ncols==0: raise ValueError("Empty data.")

                # sampling
                if tcol>0 and tcol<=ncols:
                    fs = estimate_fs(df.iloc[:, tcol-1].to_numpy(dtype=float), tunit)
                    chans = [i for i in range(ncols) if i != tcol-1]
                else:
                    fs = hz if hz>0 else 100.0
                    chans = list(range(ncols))

                # group by 6 per page
                def chunks(lst, size):
                    for i in range(0,len(lst),size): yield lst[i:i+size]
                groups=list(chunks(chans,6)); total=len(groups)

                for gi, group in enumerate(groups, 1):
                    rows = 2 if len(group)>3 else 1; cols = min(3, len(group))
                    fig, axes = plt.subplots(rows, cols, figsize=(14.8, 9.6), constrained_layout=True, sharex=True)
                    if rows==1 and cols==1: axes=np.array([[axes]])
                    elif rows==1: axes=np.array([axes])
                    elif cols==1: axes=np.array([[ax] for ax in axes])
                    axs = axes.ravel()
                    mappable=None; zlabel=""

                    for si, col in enumerate(group):
                        ax = axs[si]
                        y = df.iloc[:, col].to_numpy(dtype=float)
                        if remove_dc: y = y - np.nanmean(y)
                        if hp>0 or lp>0: y = butter_zero(y, fs, hp, lp)

                        nper = max(16, min(nperseg, len(y)))
                        nov  = min(max(0, noverlap), nper-1)
                        nfft_eff = max(nper, nfft)

                        # ---- Matlab-like spectrogram core ----
                        f, t_vec, Spec_lin = self.compute_spectrogram_like_matlab(
                            y, fs, window, nper, nov, nfft_eff, mode
                        )

                        # frequency band limit like Matlab's 'f' argument
                        fmax_eff = fs/2.0 if fmax<=0 else min(fmax, fs/2.0)
                        fmin_eff = max(0.0, fmin)
                        band = (f>=fmin_eff) & (f<=fmax_eff)
                        if band.sum()==0: band[:] = True
                        f_disp = f[band]
                        Spec_lin = Spec_lin[band, :]

                        # to dB if requested (magnitude -> 20log10, psd -> 10log10)
                        if use_db:
                            if mode == "psd":
                                Spec = 10*np.log10(Spec_lin + np.finfo(float).eps)
                                zlabel = "PSD (dB)"
                            else:
                                Spec = 20*np.log10(Spec_lin + np.finfo(float).eps)
                                zlabel = "Magnitude (dB)"
                        else:
                            Spec = Spec_lin
                            zlabel = "PSD" if mode=="psd" else "Magnitude"

                        # robust clim
                        if use_db:
                            vmin, vmax = choose_clim_db(Spec, dr=dr, vmax_pct=vmax_pct)
                        else:
                            vmin = np.nanpercentile(Spec, 5.0); vmax = np.nanpercentile(Spec, 99.0)
                            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax<=vmin:
                                vmin, vmax = None, None

                        # render
                        if render.startswith("imshow"):
                            extent=[t_vec.min(), t_vec.max(), f_disp.min(), f_disp.max()]
                            interp = "nearest" if "nearest" in render else ("bilinear" if "bilinear" in render else "bicubic")
                            im = ax.imshow(Spec, origin="lower", aspect="auto", extent=extent,
                                           cmap=cmap, interpolation=interp, vmin=vmin, vmax=vmax)
                        elif render=="pcolormesh-nearest":
                            im = ax.pcolormesh(t_vec, f_disp, Spec, shading="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
                        else:
                            im = ax.pcolormesh(t_vec, f_disp, Spec, shading="gouraud", cmap=cmap, vmin=vmin, vmax=vmax)

                        title = names[col] if col < len(names) else f"col{col+1}"
                        ax.set_title(str(title), fontsize=11)
                        if si//cols == rows-1: ax.set_xlabel("Time (s)")
                        else: ax.tick_params(labelbottom=False)
                        if si%cols == 0: ax.set_ylabel("Freq (Hz)")
                        else: ax.tick_params(labelleft=False)
                        mappable = im

                    for j in range(len(group), rows*cols): axs[j].axis("off")
                    if mappable is not None:
                        cbar = fig.colorbar(mappable, ax=axs[:len(group)], shrink=0.92); cbar.set_label(zlabel)

                    fig.suptitle(f"{Path(path).name} | {window} nperseg={nperseg} noverlap={noverlap} nfft={nfft}", fontsize=13)
                    tab = ttk.Frame(self.tabs)
                    self.tabs.add(tab, text=Path(path).name[-24:] + (f" ({gi}/{total})" if total>1 else ""))
                    canvas = FigureCanvasTkAgg(fig, master=tab); canvas.draw()
                    canvas.get_tk_widget().pack(fill="both", expand=True); NavigationToolbar2Tk(canvas, tab)

            except Exception as e:
                messagebox.showerror("Error", f"Failed: {path}\n\n{e}")

if __name__ == "__main__":
    App().mainloop()
