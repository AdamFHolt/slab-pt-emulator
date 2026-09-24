#!/usr/bin/env python3
"""
Emulator validation + Sobol sensitivity summary for one suite (three panels).

(A) Held-out accuracy of the profile-PCA + GP emulator (gp_m25) at 5 Myr:
    emulated vs. ASPECT slab-top T for every validation run at every depth of
    the PCA grid (coloured by depth), against the 1:1 line, pooled RMSE printed.
    Reconstructed from the saved model's yhat_val.npy with the same math as
    evaluate_profile_pca_quality.py (read-only, nothing re-fit) and asserted
    equal to the stored RMSE.
(B) Held-out profile RMSE vs. depth at 0.5, 3, 5 and 10 Myr (plasma time
    colours, the same time -> colour recipe as the other repo figures). Times
    whose model folder does not exist are skipped.
(C) Sobol total-effect index S_T vs. depth (10-80 km every 5 km) for the five
    design parameters, from the single-depth dTdt emulators, for the 0.5-5 Myr
    window (solid, `sobol/`) and the 5-10 Myr window (dashed, `sobol_dt10-20/`).
    A window whose directory is missing is skipped.

Copied into the repo 2026-09-24 from the NSF proposal's Figure 3
(nsf-slab-pt/figures/scripts/fig03_emulator_sobol.py, 2026-09-07) and
generalised to a SUITE argument. Style unchanged: Myriad Pro (falls back to
Arial/DejaVu), 6.5 x 2.55 in, fonts embedded as text.

Inputs (all under the repo root):
  src/emulator/models/<PCA_SET>/<suite>/runs/profileT_pca_t{T}Myr_k10/gp_m25/
      {profile_pca_quality.json, yhat_val.npy}
  src/emulator/data/<PCA_SET>/<suite>/runs/profileT_pca_t{T}Myr_k10/
      {metadata.json, val_idx.npy, pca_*.npy}   (metadata points at the Tprof CSVs)
  plots/science-emulator/single_depth/<suite>/{sobol,sobol_dt10-20}/{d}km_dTdt_gp_m25_sobol.json
  PCA_SET is profile_pca_10myr when that exists for the suite (const-vc,
  0.5-10 Myr), otherwise profile_pca (0.5-5 Myr only).

Usage (repo root, repo env):
  env/bin/python src/emulator/science/plot_emulator_validation_sobol.py SUITE [--pca-set NAME]
Output: plots/science-emulator/summary/<suite>/emulator_validation_sobol_<suite>.{pdf,svg,png}
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("suite", nargs="?", default="const-vc")
ap.add_argument("--pca-set", default=None, help="profile_pca_10myr | profile_pca (default: auto)")
ap.add_argument("--model-tag", default="gp_m25")
ap.add_argument("--out-dir", default=None)
args = ap.parse_args()
SUITE, TAG = args.suite, args.model_tag

if args.pca_set is None:
    cand = os.path.join(ROOT, "src", "emulator", "models", "profile_pca_10myr", SUITE, "runs")
    PCA_SET = "profile_pca_10myr" if os.path.isdir(cand) else "profile_pca"
else:
    PCA_SET = args.pca_set
PROFILE_MODEL_ROOT = os.path.join(ROOT, "src", "emulator", "models", PCA_SET, SUITE, "runs")
PROFILE_DATA_ROOT = os.path.join(ROOT, "src", "emulator", "data", PCA_SET, SUITE, "runs")
SOBOL_ROOT = os.path.join(ROOT, "plots", "science-emulator", "single_depth", SUITE)
OUT_DIR = args.out_dir or os.path.join(ROOT, "plots", "science-emulator", "summary", SUITE)
os.makedirs(OUT_DIR, exist_ok=True)
print(f"suite {SUITE}: PCA set {PCA_SET}, model {TAG}")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Myriad Pro", "Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Myriad Pro",
    "mathtext.it": "Myriad Pro:italic",
    "mathtext.bf": "Myriad Pro:bold",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.linewidth": 0.7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


# ========================================================================
# Panel (A): held-out profile-PCA + GP reconstruction at 5 Myr
# ========================================================================
def load_profile_on_grid(csv_path, depth_grid):
    """Port of evaluate_profile_pca_quality.py's _load_profile_on_grid."""
    df = pd.read_csv(csv_path)
    z = pd.to_numeric(df["depth_km"], errors="coerce").to_numpy(float)
    t = pd.to_numeric(df["T_C"], errors="coerce").to_numpy(float)
    mask = np.isfinite(z) & np.isfinite(t)
    z, t = z[mask], t[mask]
    order = np.argsort(z)
    z, t = z[order], t[order]
    z_u, idx = np.unique(z, return_index=True)
    out = np.interp(depth_grid, z_u, t[idx], left=np.nan, right=np.nan)
    if not np.isfinite(out).all():
        raise ValueError(f"{csv_path} does not fully cover depth grid")
    return out


def reconstruct_val_profiles(time_tag):
    dataset_dir = os.path.join(PROFILE_DATA_ROOT, f"profileT_pca_{time_tag}_k10")
    model_dir = os.path.join(PROFILE_MODEL_ROOT, f"profileT_pca_{time_tag}_k10", TAG)
    with open(os.path.join(dataset_dir, "metadata.json")) as f:
        meta = json.load(f)
    depth_grid = np.asarray(meta["profile"]["depth_grid_km"], dtype=float)
    source_paths = list(meta["profile"]["source_paths"])
    val_idx = np.load(os.path.join(dataset_dir, "val_idx.npy"))
    mean_profile = np.load(os.path.join(dataset_dir, "pca_mean_profile.npy"))
    components = np.load(os.path.join(dataset_dir, "pca_components.npy"))
    score_scale = np.load(os.path.join(dataset_dir, "pca_score_scale.npy"))
    score_space = str(meta.get("target", {}).get("score_space", "raw")).strip().lower()
    true_profiles = np.vstack([load_profile_on_grid(p, depth_grid) for p in source_paths])
    yhat_val = np.load(os.path.join(model_dir, "yhat_val.npy"))
    pred_scores_raw = yhat_val * score_scale[None, :] if score_space == "whitened" else yhat_val
    emu_recon_val = mean_profile[None, :] + pred_scores_raw @ components
    return depth_grid, true_profiles[val_idx], emu_recon_val


def load_quality_rmse_by_depth(time_tag):
    qpath = os.path.join(PROFILE_MODEL_ROOT, f"profileT_pca_{time_tag}_k10", TAG, "profile_pca_quality.json")
    with open(qpath) as f:
        q = json.load(f)
    depth_grid = np.asarray(q["depth_grid_km"], dtype=float)
    rec = q["metrics"]["val"]["profile_space"]["emulator_reconstruction"]
    return depth_grid, np.asarray(rec["rmse_by_depth"], dtype=float), rec["rmse"], q["metrics"]["val"]["n_rows"]


def has_time(time_tag):
    return os.path.isfile(os.path.join(PROFILE_MODEL_ROOT, f"profileT_pca_{time_tag}_k10", TAG,
                                       "profile_pca_quality.json"))


A_TAG = "t5Myr"
if not has_time(A_TAG):
    raise SystemExit(f"no {TAG} model for {A_TAG} under {PROFILE_MODEL_ROOT}")
depth_grid_A, true_val_A, emu_val_A = reconstruct_val_profiles(A_TAG)
rmse_A = float(np.sqrt(np.mean((true_val_A - emu_val_A) ** 2)))
n_val_A = true_val_A.shape[0]
print(f"Panel A: t=5 Myr, n_val_runs={n_val_A}, n_depth={true_val_A.shape[1]}, pooled RMSE = {rmse_A:.2f} degC")
_, _, rmse_A_stored, n_val_stored = load_quality_rmse_by_depth(A_TAG)
print(f"  cross-check vs. profile_pca_quality.json: stored RMSE = {rmse_A_stored:.6f} (n_val={n_val_stored})")
assert abs(rmse_A - rmse_A_stored) < 1e-6, "reconstruction does not match stored RMSE"

# ========================================================================
# Panel (B): held-out profile RMSE vs depth at up to four times
# ========================================================================
k_list_full = [1, 2, 4, 6, 10, 15, 20]
plasma = plt.get_cmap("plasma")
k_to_color = {k: plasma(s) for k, s in zip(k_list_full, np.linspace(0.08, 0.82, len(k_list_full)))}

B_TIMES = [("t0p5Myr", 0.5, 1), ("t3Myr", 3.0, 6), ("t5Myr", 5.0, 10), ("t10Myr", 10.0, 20)]
panelB_data = []
for tag, t_myr, k in B_TIMES:
    if not has_time(tag):
        print(f"Panel B: t={t_myr:g} Myr  -- no {TAG} model, skipped")
        continue
    depth_grid_b, rmse_by_depth, rmse_overall, n_val = load_quality_rmse_by_depth(tag)
    panelB_data.append((t_myr, k, depth_grid_b, rmse_by_depth, rmse_overall, n_val))
    print(f"Panel B: t={t_myr:g} Myr  n_val={n_val}  RMSE(depth) range = "
          f"{rmse_by_depth.min():.2f}-{rmse_by_depth.max():.2f} degC (pooled {rmse_overall:.2f})")

# ========================================================================
# Panel (C): Sobol total-effect index vs depth, up to two time windows
# ========================================================================
PARAMS = ["age_OP", "v_conv", "age_SP", "dip_int", "eta_UM"]
PARAM_LABEL = {"v_conv": r"$v_{\mathrm{conv}}$", "age_SP": r"$\mathrm{age}_{\mathrm{SP}}$",
               "age_OP": r"$\mathrm{age}_{\mathrm{OP}}$", "dip_int": r"$\theta_{\mathrm{slab}}$",
               "eta_UM": r"$\eta_{\mathrm{UM}}$"}
PARAM_COLOR = {"age_OP": "#009E73", "v_conv": "#0072B2", "age_SP": "#D55E00",
               "dip_int": "#E69F00", "eta_UM": "#CC79A7"}   # Okabe-Ito
DEPTHS_C = list(range(10, 81, 5))
WINDOWS_ALL = [("0.5-5 Myr", os.path.join(SOBOL_ROOT, "sobol"), "-", "o"),
               ("5-10 Myr", os.path.join(SOBOL_ROOT, "sobol_dt10-20"), "--", "s")]
WINDOWS = []
sobol_ST = {}
for label, sdir, ls, mk in WINDOWS_ALL:
    if not os.path.isdir(sdir):
        print(f"Panel C: window {label} -- {sdir} missing, skipped")
        continue
    per_param = {p: [] for p in PARAMS}
    for d in DEPTHS_C:
        with open(os.path.join(sdir, f"{d}km_dTdt_{TAG}_sobol.json")) as f:
            js = json.load(f)
        cols = list(js["feature_cols"])
        for p in PARAMS:
            per_param[p].append(js["ST"][cols.index(p)])
    sobol_ST[label] = {p: np.asarray(v) for p, v in per_param.items()}
    WINDOWS.append((label, sdir, ls, mk))
if not WINDOWS:
    raise SystemExit(f"no Sobol directories under {SOBOL_ROOT}")

depths_arr = np.asarray(DEPTHS_C, dtype=float)
for label, *_ in WINDOWS:
    diff = sobol_ST[label]["age_OP"] - sobol_ST[label]["v_conv"]
    crossings = []
    for i in range(len(depths_arr) - 1):
        if np.sign(diff[i]) != 0 and np.sign(diff[i]) != np.sign(diff[i + 1]) and diff[i] != diff[i + 1]:
            d0, d1 = depths_arr[i], depths_arr[i + 1]
            crossings.append(d0 + (d1 - d0) * (-diff[i]) / (diff[i + 1] - diff[i]))
    if crossings:
        print(f"Panel C: window {label}, age_OP/v_conv ST crossover(s) at "
              f"{', '.join(f'{z:.1f}' for z in crossings)} km")
    else:
        print(f"Panel C: window {label}, no age_OP/v_conv crossover in 10-80 km "
              f"({'age_OP' if diff[0] > 0 else 'v_conv'} dominates throughout)")

# ========================================================================
# Layout
# ========================================================================
# Legends of (C) live OUTSIDE the panel, in a strip to its right (LEGEND_W), because for ramped-vc
# the curves fill the whole panel and any in-panel box hides data (PI, 2026-09-24). Panel geometry is
# the proposal's 6.5 in layout; the strip is added to the figure width.
LEGEND_W = 0.62
FIG_W, FIG_H = 6.5 + LEGEND_W, 2.55
left_margin, right_margin, top_margin, bottom_margin, gap = 0.50, 0.10 + LEGEND_W, 0.20, 0.42, 0.55
panel_h = FIG_H - top_margin - bottom_margin
panel_wA = panel_h
panel_w = (FIG_W - left_margin - right_margin - 2 * gap - panel_wA) / 2.0
fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")


def axes_in(x0_in, y0_in, w_in, h_in):
    return fig.add_axes([x0_in / FIG_W, y0_in / FIG_H, w_in / FIG_W, h_in / FIG_H])


y0 = bottom_margin
xA = left_margin
xB = xA + panel_wA + gap
xC = xB + panel_w + gap
axA = axes_in(xA, y0, panel_wA, panel_h)
axB = axes_in(xB, y0, panel_w, panel_h)
axC = axes_in(xC, y0, panel_w, panel_h)

zmax = float(depth_grid_A.max())
# ---------------- Panel A ----------------
sc = axA.scatter(true_val_A.ravel(), emu_val_A.ravel(), c=np.tile(depth_grid_A, n_val_A),
                 cmap=plt.get_cmap("viridis"), vmin=0, vmax=zmax, s=3.0, linewidths=0, alpha=0.75)
lims = [0, 1200]
axA.plot(lims, lims, color="0.35", lw=0.8, zorder=1)
axA.set_xlim(lims); axA.set_ylim(lims)
axA.set_xticks(np.arange(0, 1201, 400)); axA.set_yticks(np.arange(0, 1201, 400))
axA.set_aspect("equal", adjustable="box", anchor="N")
axA.set_xlabel("Modelled slab-top T (" + r"$^{\circ}$" + "C)")
axA.set_ylabel("Emulated slab-top T (" + r"$^{\circ}$" + "C)")
axA.tick_params(labelsize=7)
axA.text(0.05, 0.92, f"5 Myr\nRMSE = {rmse_A:.1f}" + r"$^{\circ}$" + f"C\n(n={n_val_A} held-out)",
         transform=axA.transAxes, fontsize=7, ha="left", va="top")
cax_a = axA.inset_axes([0.66, 0.08, 0.05, 0.34])
cb_a = fig.colorbar(sc, cax=cax_a)
cb_a.set_label("Depth (km)", fontsize=7, labelpad=2)
cb_a.set_ticks([0, zmax / 2, zmax])
cb_a.ax.tick_params(labelsize=7, length=2, pad=1.5)
cb_a.outline.set_linewidth(0.5)

# ---------------- Panel B ----------------
for t_myr, k, depth_grid_b, rmse_by_depth, rmse_overall, n_val in panelB_data:
    axB.plot(rmse_by_depth, depth_grid_b, color=k_to_color[k], lw=1.3, label=f"{t_myr:g} Myr")
axB.set_ylim(zmax, 0)
axB.set_xlim(left=0)
axB.set_yticks(np.arange(0, zmax + 1, 20))
axB.set_xlabel("Profile RMSE (" + r"$^{\circ}$" + "C)")
axB.set_ylabel("Depth (km)")
axB.tick_params(labelsize=7)
axB.spines["top"].set_visible(False); axB.spines["right"].set_visible(False)
leg_b = axB.legend(fontsize=7, loc="upper right", frameon=True, facecolor="white", edgecolor="black",
                   framealpha=0.9, handlelength=1.8, labelspacing=0.25, borderpad=0.35,
                   title="time since\ninitiation", title_fontsize=7)
leg_b.get_frame().set_linewidth(0.5)

# ---------------- Panel C ----------------
for label, _, ls, mk in WINDOWS:
    for p in PARAMS:
        axC.plot(sobol_ST[label][p], depths_arr, color=PARAM_COLOR[p], ls=ls, lw=1.3, marker=mk,
                 ms=2.2 if ls == "-" else 2.6, mfc=PARAM_COLOR[p] if ls == "-" else "white", mew=0.7)
axC.set_ylim(80, 0)
axC.set_xlim(0, 1.0)
axC.set_yticks(np.arange(0, 81, 20)); axC.set_xticks([0, 0.5, 1.0])
axC.set_xlabel(r"Sobol $S_T$")
axC.set_ylabel("Depth (km)")
axC.tick_params(labelsize=7)
axC.spines["top"].set_visible(False); axC.spines["right"].set_visible(False)

param_handles = [Line2D([0], [0], color=PARAM_COLOR[p], lw=1.3, label=PARAM_LABEL[p]) for p in PARAMS]
window_handles = [Line2D([0], [0], color="0.25", lw=1.3, ls=ls, marker=mk, ms=2.2 if ls == "-" else 2.6,
                         mfc="0.25" if ls == "-" else "white", mew=0.7, label=label)
                  for label, _, ls, mk in WINDOWS]
# both keys outside the panel, stacked in the right-hand strip, bottom-aligned with the axes
leg_c1 = axC.legend(handles=param_handles, fontsize=6.5, loc="lower left",
                    bbox_to_anchor=(1.04, 0.30), bbox_transform=axC.transAxes, frameon=True,
                    facecolor="white", edgecolor="black", framealpha=0.95, handlelength=1.0,
                    handletextpad=0.5, labelspacing=0.2, borderpad=0.3, borderaxespad=0)
leg_c1.get_frame().set_linewidth(0.5)
axC.add_artist(leg_c1)
leg_c2 = axC.legend(handles=window_handles, fontsize=6.5, loc="lower left",
                    bbox_to_anchor=(1.04, 0.0), bbox_transform=axC.transAxes, frameon=True,
                    facecolor="white", edgecolor="black", framealpha=0.95, handlelength=1.8,
                    labelspacing=0.25, borderpad=0.35, borderaxespad=0)
leg_c2.get_frame().set_linewidth(0.5)

fig.canvas.draw()
for ax, label in [(axA, "(A)"), (axB, "(B)"), (axC, "(C)")]:
    bbox = ax.get_position()
    fig.text(bbox.x0 - 0.010, bbox.y1 + 0.012, label, fontsize=9, fontweight="bold", ha="left", va="bottom")
fig.text(0.995, 0.985, SUITE, fontsize=7, color="0.4", ha="right", va="top")

base = os.path.join(OUT_DIR, f"emulator_validation_sobol_{SUITE}")
for ext in ("pdf", "svg"):
    fig.savefig(f"{base}.{ext}", facecolor="white")
fig.savefig(f"{base}.png", facecolor="white", dpi=300)
print("wrote", base + ".{pdf,svg,png}")
