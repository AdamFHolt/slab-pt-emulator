#!/usr/bin/env python3
"""Build the const-vc-sh suite: const-vc with shear heating switched on (ASPECT >= 3.0).

const-vc-sh is a one-term extension of const-vc: the viscous dissipation
2 eta (eps_dev : eps_dev) is added to the temperature equation.  Everything
else -- design (same 400-point Latin hypercube; data/params/params-list.const-vc-sh.*
are copies of the const-vc lists), geometry, rheology, boundary conditions,
solver settings, output cadence -- is identical, so run_XXX pairs with
const-vc run_XXX.

Edits to each const-vc .prm (reference template
data/ref-model/model_template_fixed-trench.const-vc-sh.prm, kept in step):
  1. Formulation: 'Boussinesq approximation' -> 'custom' with
     'Mass conservation = incompressible' and
     'Temperature equation = reference density profile'.  These are the two
     choices the Boussinesq option makes internally; ASPECT aborts if the
     shear-heating plugin is combined with the Boussinesq keyword, and the
     custom formulation is the documented route for "shear heating without
     adiabatic heating".
  2. Heating model: 'List of model names = shear heating', with the plugin's
     stress limiter on (Drucker-Prager cap, cohesion 10 MPa, friction angle
     0.523599 rad = 30 deg: the mantle plasticity values).  Limiter parameters
     exist from ASPECT 3.0 on.
  3. 'Stokes solver type = block AMG' made explicit: it was the 2.5 default that
     const-vc ran with, but 3.0 defaults to GMG.
  4. Visualization output gains 'heating' (W/m^3) next to 'viscosity'.

The suite is meant to run under ASPECT 3.0 / 3.x-dev while const-vc ran under
2.5, so a version-control set is built alongside (--control): the pilot-list
const-vc .prm files with ONLY edit 3 applied, under
subd-model-runs/const-vc-v3ctrl/run-inputs/.  Running those under 3.x and
comparing with the existing const-vc outputs isolates the version effect from
the heating effect.

Like build_runs.const-vc-dd100.py this derives each run from the existing
const-vc run-inputs (plate input files hard-linked, ~0 extra disk; rsync to TACC
with `make push-tacc SUITE=const-vc-sh LINK=const-vc-new`) and asserts the
body of each .prm changed by exactly the expected lines.  Scheduler scripts are
copied from const-vc-dd100/run-inputs with the suite name swapped; run_one.slurm
still loads the 2.5 toolchain and carries a TODO for the 3.x binary.

Usage:  python src/build-numerical-mods/build_runs.const-vc-sh.py [--control]
Reconciliation of the collaborator's test file (subd-model-runs/const-vc-sh/example/)
with the production settings: subd-model-runs/const-vc-sh/README.md.
"""
import os, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_SUITE, NEW_SUITE = "const-vc", "const-vc-sh"
SRC_DIR = ROOT / "subd-model-runs" / SRC_SUITE / "run-inputs"
SCRIPT_SRC_DIR = ROOT / "subd-model-runs" / "const-vc-dd100" / "run-inputs"
OUT_DIR = ROOT / "subd-model-runs" / NEW_SUITE / "run-inputs"
CTRL_SUITE = "const-vc-v3ctrl"
CTRL_DIR = ROOT / "subd-model-runs" / CTRL_SUITE / "run-inputs"
PILOT_LIST = OUT_DIR / "pilot-list.txt"
CONTROL = "--control" in sys.argv[1:]
TEMPLATE = ROOT / "data" / "ref-model" / "model_template_fixed-trench.const-vc-sh.prm"

OLD_FORM = """subsection Formulation
  set Formulation = Boussinesq approximation
end
"""
NEW_FORM = """# const-vc-sh: shear heating on.  ASPECT's 'Boussinesq approximation' formulation
# refuses the shear-heating plugin, so the same equations are requested piecewise:
# incompressible mass conservation + reference-density-profile temperature equation
# (exactly what the Boussinesq option selects), with the heating models listed
# explicitly.  No 'adiabatic heating' -> the only new term is the dissipation.
subsection Formulation
  set Formulation = custom
  set Mass conservation = incompressible
  set Temperature equation = reference density profile
end

# Viscous dissipation 2 eta (eps - tr(eps)/3 I):(eps - tr(eps)/3 I), eta and eps from the
# material model / solution.  The stress entering the heating is capped at a Drucker-Prager
# yield stress with the same cohesion / friction angle as the mantle plasticity
# (10 MPa, 30 deg; the plugin takes the angle in RADIANS, unlike the material model).
# Requires ASPECT >= 3.0 (parameters absent in 2.5).
subsection Heating model
  set List of model names = shear heating
  subsection Shear heating
    set Limit stress contribution to shear heating = true
    set Cohesion for maximum shear stress = 10e6
    set Friction angle for maximum shear stress = 0.523599
  end
end
"""
OLD_AMG = """  subsection Stokes solver parameters
    set Linear solver tolerance  		= 1.0e-4
"""
NEW_AMG = """  subsection Stokes solver parameters
    set Stokes solver type = block AMG   # const-vc-sh: 2.5 default, made explicit because 3.0 defaults to GMG
    set Linear solver tolerance  		= 1.0e-4
"""
OLD_VIS = "    set List of output variables = viscosity\n"
NEW_VIS = "    set List of output variables = viscosity, heating\n"
OLD_BASE, NEW_BASE = "production-runs_v2/const-vc-dd100", f"production-runs_v2/{NEW_SUITE}"
SCRIPTS = ["run_one.slurm", "submit_from_list.sh", "submit_rolling.sh"]

# the template must carry exactly the same edits, so the record stays in step
tmpl = TEMPLATE.read_text()
for frag in (NEW_FORM, NEW_VIS, NEW_AMG):
    assert tmpl.count(frag) == 1, f"template out of step with this script: {frag[:40]!r}"

if not SRC_DIR.is_dir():
    sys.exit(f"const-vc run-inputs not found at {SRC_DIR}; regenerate them first "
             "(build_runs.const-vc.py) or adapt this script to the template.")
OUT_DIR.mkdir(parents=True, exist_ok=True)

n_extra_lines = NEW_FORM.count("\n") - OLD_FORM.count("\n")
runs = sorted(p for p in SRC_DIR.glob("run_*") if p.is_dir())
n_ok = 0
for src_run in runs:
    name = src_run.name
    src_prm = src_run / f"{name}.prm"
    if not src_prm.is_file():
        print(f"  skip {name}: no {src_prm.name}")
        continue
    text = src_prm.read_text()
    for frag in (OLD_FORM, OLD_VIS, OLD_AMG):
        if text.count(frag) != 1:
            sys.exit(f"{src_prm}: expected exactly one occurrence of {frag[:40]!r}, found {text.count(frag)}")
    new = text.replace(OLD_FORM, NEW_FORM).replace(OLD_VIS, NEW_VIS).replace(OLD_AMG, NEW_AMG)
    new = new.replace("# This file is auto-generated by build_runs.const-vc.py",
                      "# This file is auto-generated by build_runs.const-vc-sh.py (target: ASPECT >= 3.0)\n"
                      f"# Derived from {SRC_SUITE}/run-inputs/{name}/{name}.prm: identical except\n"
                      "# Formulation -> custom (incompressible, reference density profile),\n"
                      "# Heating model = shear heating (stress-limited, 10 MPa / 30 deg),\n"
                      "# Stokes solver type = block AMG made explicit, 'heating' added to the output variables.")

    body_old = text.split("\nset Dimension", 1)[1].splitlines()
    body_new = new.split("\nset Dimension", 1)[1].splitlines()
    assert len(body_new) == len(body_old) + n_extra_lines + 1, f"{name}: unexpected line count"
    # every line of the original body must survive except the two replaced ones
    removed = [l for l in body_old if l not in body_new]
    assert removed == ["  set Formulation = Boussinesq approximation", OLD_VIS.rstrip("\n")], \
        f"{name}: unexpected removed lines {removed}"

    dst_run = OUT_DIR / name
    (dst_run / "inputs").mkdir(parents=True, exist_ok=True)
    (dst_run / f"{name}.prm").write_text(new)
    for f in sorted((src_run / "inputs").iterdir()):
        dst = dst_run / "inputs" / f.name
        if dst.exists():
            continue
        try:
            os.link(f, dst)
        except OSError:
            shutil.copy2(f, dst)
    n_ok += 1

def copy_scripts(dst_dir: Path, new_base: str) -> None:
    for sname in SCRIPTS:
        src = SCRIPT_SRC_DIR / sname
        if not src.is_file():
            continue
        t = src.read_text()
        assert OLD_BASE in t, f"{sname}: BASE_DIR pattern not found"
        t = t.replace(OLD_BASE, new_base)
        if sname == "run_one.slurm":
            t = t.replace('ASPECT_EXE="${asp25_skx:?asp25_skx not set}"',
                          '# TODO(const-vc-sh): this suite needs an ASPECT >= 3.0 binary (stress-limited shear heating);\n'
                          '# swap the module line above and the variable below for the 3.x build before submitting.\n'
                          'ASPECT_EXE="${asp30_skx:?asp30_skx not set (ASPECT 3.x build, see const-vc-sh/README.md)}"')
        (dst_dir / sname).write_text(t)
        shutil.copymode(src, dst_dir / sname)

copy_scripts(OUT_DIR, NEW_BASE)

n_ctrl = 0
if CONTROL:
    # version-control set: pilot-list const-vc runs with only the solver line made explicit
    pilot = [l.strip().zfill(3) for l in PILOT_LIST.read_text().split() if l.strip()]
    CTRL_DIR.mkdir(parents=True, exist_ok=True)
    for rid in pilot:
        name = f"run_{rid}"
        text = (SRC_DIR / name / f"{name}.prm").read_text()
        assert text.count(OLD_AMG) == 1
        new = text.replace(OLD_AMG, NEW_AMG).replace(
            "# This file is auto-generated by build_runs.const-vc.py",
            "# This file is auto-generated by build_runs.const-vc-sh.py --control (target: ASPECT >= 3.0)\n"
            f"# VERSION CONTROL: {SRC_SUITE}/run-inputs/{name}/{name}.prm with only 'Stokes solver type = block AMG'\n"
            "# made explicit (3.0 defaults to GMG).  No heating.  Compare with the existing const-vc (2.5) output.")
        dst_run = CTRL_DIR / name
        (dst_run / "inputs").mkdir(parents=True, exist_ok=True)
        (dst_run / f"{name}.prm").write_text(new)
        for f in sorted((SRC_DIR / name / "inputs").iterdir()):
            dst = dst_run / "inputs" / f.name
            if not dst.exists():
                try:
                    os.link(f, dst)
                except OSError:
                    shutil.copy2(f, dst)
        n_ctrl += 1
    copy_scripts(CTRL_DIR, f"production-runs_v2/{CTRL_SUITE}")
    shutil.copy2(PILOT_LIST, CTRL_DIR / "pilot-list.txt")

print(f"Finished: {n_ok} run directories under {OUT_DIR} "
      f"(prm: Formulation custom + stress-limited shear heating + block AMG explicit + 'heating' output; "
      f"inputs hard-linked from {SRC_SUITE}); scripts: {', '.join(SCRIPTS)} with BASE_DIR -> {NEW_BASE}"
      + (f"; version-control set: {n_ctrl} runs under {CTRL_DIR}" if CONTROL else ""))
