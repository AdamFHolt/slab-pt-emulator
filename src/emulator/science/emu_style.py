"""Shared style and loaders for the emulator science figures (house style of the NSF proposal
figures: Myriad Pro incl. mathtext, 8 pt, fonts embedded as text in PDF/SVG).

Conventions used by plot_emulator_validation.py and plot_sobol_windows.py:
  * suite  -> line style + marker fill (const-vc solid/filled, ramped-vc dashed/hollow; suite_kw());
              marker shape never encodes the suite
  * time   -> plasma colour (the k -> colour recipe of the proposal's Fig. 1D / Fig. 3B)
  * window -> three fixed colours; parameter -> Okabe-Ito colours (fixed per parameter)
"""
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
OUT_DIR = os.path.join(ROOT, "plots", "science-emulator", "summary")

RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Myriad Pro", "Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Myriad Pro",
    "mathtext.it": "Myriad Pro:italic",
    "mathtext.bf": "Myriad Pro:bold",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.linewidth": 0.7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}


def apply_style():
    plt.rcParams.update(RC)


DEG = r"$^{\circ}$"
SUITE_LS = {"const-vc": "-", "ramped-vc": "--", "const-vc-sh": "-.", "const-vc-dd100": ":"}
SUITE_LABEL = {"const-vc": "const-vc", "ramped-vc": "ramped-vc", "const-vc-sh": "const-vc-sh",
               "const-vc-dd100": "const-vc-dd100"}

def suite_kw(suite, color, marker="o", lw=1.3):
    """Line/marker kwargs encoding the suite: const-vc solid + filled, ramped-vc dashed + hollow, ...
    Marker SHAPE never encodes the suite (it is free for windows etc.)."""
    ls = SUITE_LS.get(suite, ":")
    filled = ls == "-"
    return dict(color=color, ls=ls, lw=lw, marker=marker, ms=2.4 if filled else 2.8,
                mfc=color if filled else "white", mec=color, mew=0.7)


def suite_handles(suites, color="0.15", marker="o"):
    from matplotlib.lines import Line2D
    return [Line2D([0], [0], label=SUITE_LABEL.get(s, s), **suite_kw(s, color, marker)) for s in suites]


K_LIST_FULL = [1, 2, 4, 6, 10, 15, 20]          # 0.5, 1, 2, 3, 5, 7.5, 10 Myr
_plasma = plt.get_cmap("plasma")
K_TO_COLOR = {k: _plasma(s) for k, s in zip(K_LIST_FULL, np.linspace(0.08, 0.82, len(K_LIST_FULL)))}


def time_color(t_myr):
    """Colour for a time since initiation: nearest k in the fixed recipe."""
    k = int(round(t_myr * 2))
    if k in K_TO_COLOR:
        return K_TO_COLOR[k]
    ks = np.array(K_LIST_FULL)
    return K_TO_COLOR[int(ks[np.argmin(np.abs(ks - k))])]


PARAMS = ["age_OP", "v_conv", "age_SP", "dip_int", "eta_UM", "t_conv"]
PARAM_LABEL = {
    "v_conv": r"$v_{\mathrm{conv}}$", "age_SP": r"$\mathrm{age}_{\mathrm{SP}}$",
    "age_OP": r"$\mathrm{age}_{\mathrm{OP}}$", "dip_int": r"$\theta_{\mathrm{slab}}$",
    "eta_UM": r"$\eta_{\mathrm{UM}}$", "t_conv": r"$t_{\mathrm{ramp}}$",
}
PARAM_COLOR = {   # Okabe-Ito
    "age_OP": "#009E73", "v_conv": "#0072B2", "age_SP": "#D55E00",
    "dip_int": "#E69F00", "eta_UM": "#CC79A7", "t_conv": "#56B4E9",
}

WINDOWS = [("0.5-5 Myr", "sobol"), ("0.5-10 Myr", "sobol_dt1-20"), ("5-10 Myr", "sobol_dt10-20")]
WINDOW_COLOR = {"0.5-5 Myr": _plasma(0.15), "0.5-10 Myr": _plasma(0.50), "5-10 Myr": _plasma(0.82)}

DEEP_END_KM = 80.0   # slab top at 85-100 km exists only after crust arrival (decision 2026-09-17)


# ---------------------------------------------------------------- profile-PCA products
def pca_set_for(suite):
    cand = os.path.join(ROOT, "src", "emulator", "models", "profile_pca_10myr", suite, "runs")
    return "profile_pca_10myr" if os.path.isdir(cand) else "profile_pca"


def pca_roots(suite, pca_set=None):
    pca_set = pca_set or pca_set_for(suite)
    return (os.path.join(ROOT, "src", "emulator", "models", pca_set, suite, "runs"),
            os.path.join(ROOT, "src", "emulator", "data", pca_set, suite, "runs"))


def time_from_tag(name):
    m = re.search(r"_t([0-9p]+)Myr", name)
    return float(m.group(1).replace("p", ".")) if m else float("nan")


def tag_from_time(t):
    return f"t{t:g}".replace(".", "p") + "Myr"


def quality_reports(suite, tag_model="gp_m25", k=10, pca_set=None):
    """{time_myr: quality dict} for every scored profile-PCA slice of the suite."""
    model_root, _ = pca_roots(suite, pca_set)
    out = {}
    for q in sorted(glob.glob(os.path.join(model_root, f"profileT_pca_t*Myr_k{k}", tag_model,
                                           "profile_pca_quality.json"))):
        with open(q) as f:
            d = json.load(f)
        out[time_from_tag(d["dataset_name"])] = d
    return dict(sorted(out.items()))


def val_block(q):
    return q["metrics"]["val"]["profile_space"]


def load_profile_on_grid(csv_path, depth_grid):
    import pandas as pd
    df = pd.read_csv(csv_path)
    z = pd.to_numeric(df["depth_km"], errors="coerce").to_numpy(float)
    t = pd.to_numeric(df["T_C"], errors="coerce").to_numpy(float)
    m = np.isfinite(z) & np.isfinite(t)
    z, t = z[m], t[m]
    o = np.argsort(z)
    z_u, idx = np.unique(z[o], return_index=True)
    out = np.interp(depth_grid, z_u, t[o][idx], left=np.nan, right=np.nan)
    if not np.isfinite(out).all():
        raise ValueError(f"{csv_path} does not cover the depth grid")
    return out


def reconstruct_val_profiles(suite, t_myr, tag_model="gp_m25", k=10, pca_set=None):
    """Held-out true and emulated profiles for one time (mirrors evaluate_profile_pca_quality.py)."""
    model_root, data_root = pca_roots(suite, pca_set)
    dname = f"profileT_pca_{tag_from_time(t_myr)}_k{k}"
    ds, md = os.path.join(data_root, dname), os.path.join(model_root, dname, tag_model)
    with open(os.path.join(ds, "metadata.json")) as f:
        meta = json.load(f)
    zg = np.asarray(meta["profile"]["depth_grid_km"], float)
    val_idx = np.load(os.path.join(ds, "val_idx.npy"))
    mean_p = np.load(os.path.join(ds, "pca_mean_profile.npy"))
    comps = np.load(os.path.join(ds, "pca_components.npy"))
    scale = np.load(os.path.join(ds, "pca_score_scale.npy"))
    whitened = str(meta.get("target", {}).get("score_space", "raw")).lower() == "whitened"
    true = np.vstack([load_profile_on_grid(p, zg) for p in meta["profile"]["source_paths"]])[val_idx]
    yhat = np.load(os.path.join(md, "yhat_val.npy"))
    emu = mean_p[None, :] + (yhat * scale[None, :] if whitened else yhat) @ comps
    return zg, true, emu


# ---------------------------------------------------------------- single-depth Sobol products
def sobol_dir(suite, subdir):
    return os.path.join(ROOT, "plots", "science-emulator", "single_depth", suite, subdir)


def sobol_table(suite, subdir, tag_model="gp_m25"):
    """{depth_km: json} for one window of one suite (empty if the directory is missing)."""
    d = sobol_dir(suite, subdir)
    out = {}
    for p in glob.glob(os.path.join(d, f"*km_dTdt_{tag_model}_sobol.json")):
        with open(p) as f:
            j = json.load(f)
        out[float(j["depth_km"])] = j
    return dict(sorted(out.items()))


def outside_legend(ax, handles, y0, **kw):
    """Legend in the strip right of `ax`, lower-left corner at axes coords (1.04, y0)."""
    base = dict(fontsize=6.5, loc="lower left", bbox_to_anchor=(1.04, y0), bbox_transform=ax.transAxes,
                frameon=True, facecolor="white", edgecolor="black", framealpha=0.95,
                handlelength=1.6, handletextpad=0.5, labelspacing=0.25, borderpad=0.35, borderaxespad=0)
    base.update(kw)
    leg = ax.legend(handles=handles, **base)
    leg.get_frame().set_linewidth(0.5)
    ax.add_artist(leg)
    return leg


def panel_labels(fig, axes_labels, dx=-0.010, dy=0.012):
    fig.canvas.draw()
    for ax, lab in axes_labels:
        b = ax.get_position()
        fig.text(b.x0 + dx, b.y1 + dy, lab, fontsize=9, fontweight="bold", ha="left", va="bottom")


def save(fig, stem):
    os.makedirs(OUT_DIR, exist_ok=True)
    base = os.path.join(OUT_DIR, stem)
    fig.savefig(base + ".pdf", facecolor="white")
    fig.savefig(base + ".svg", facecolor="white")
    fig.savefig(base + ".png", facecolor="white", dpi=300)
    print("wrote", base + ".{pdf,svg,png}")
