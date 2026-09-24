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

--mu05 builds a second 8-run pilot, subd-model-runs/const-vc-sh-mu05/run-inputs/: the
pilot-list const-vc-sh .prm files with ONLY the heating-stress cap changed to a low
effective friction, cohesion 1 MPa and friction angle asin(0.05) = 0.050021 rad
(mu' = sin(phi) = 0.05, Kohn et al. 2018's global best fit), so the dissipation
stress is min(2 eta eps, 1 MPa + 0.05 P) instead of min(2 eta eps, 10 MPa + 0.5 P).
Mechanics untouched (the cap only enters the heating term), so the runs still pair
with const-vc / const-vc-sh run_XXX.  Decided 2026-09-24; the mechanics-consistent
version (Drucker-Prager yield in the crust composition) is a later suite.

Usage:  python src/build-numerical-mods/build_runs.const-vc-sh.py [--control] [--mu05]
Reconciliation of the collaborator's test file (subd-model-runs/const-vc-sh/example/)
with the production settings: subd-model-runs/const-vc-sh/README.md.
"""
import os, re, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_SUITE, NEW_SUITE = "const-vc", "const-vc-sh"
SRC_DIR = ROOT / "subd-model-runs" / SRC_SUITE / "run-inputs"
SCRIPT_SRC_DIR = ROOT / "subd-model-runs" / "const-vc-dd100" / "run-inputs"
OUT_DIR = ROOT / "subd-model-runs" / NEW_SUITE / "run-inputs"
CTRL_SUITE = "const-vc-v3ctrl"
CTRL_DIR = ROOT / "subd-model-runs" / CTRL_SUITE / "run-inputs"
MU_SUITE = "const-vc-sh-mu05"
MU_DIR = ROOT / "subd-model-runs" / MU_SUITE / "run-inputs"
PILOT_LIST = OUT_DIR / "pilot-list.txt"
CONTROL = "--control" in sys.argv[1:]
MU05 = "--mu05" in sys.argv[1:]
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
# ASPECT 3.0 removed the standalone 'viscosity' visualization postprocessor ("has been removed,
# please use the 'material properties' postprocessor instead"); the replacement still writes a vtu
# field *named* viscosity, so the field-CSV extraction is unaffected.  This is what the
# collaborator's 3.x test file used, and keeping our 2.5 spelling aborted every run at startup
# (caught by a 48-rank smoke test, 2026-09-22).
NEW_VIS = ("    set List of output variables = material properties, heating\n"
           "    subsection Material properties\n"
           "      set List of material properties = viscosity\n"
           "    end\n")
# the control set has no heating, but still needs the 3.0 spelling to start at all
CTRL_VIS = ("    set List of output variables = material properties\n"
            "    subsection Material properties\n"
            "      set List of material properties = viscosity\n"
            "    end\n")
OLD_BASE, NEW_BASE = "production-runs_v2/const-vc-dd100", f"production-runs_v2/{NEW_SUITE}"
SCRIPTS = ["run_one.slurm", "submit_from_list.sh", "submit_rolling.sh"]
# Slurm job names: run_XXX is what the 2.5 suites use, and dd100's feeder throttles on that
# pattern across the whole account, so the 3.0 suites get their own prefixes.  The run
# DIRECTORY stays run_XXX (the batch script derives it from RUN_ID); the prefix only names the
# job and its log files.
JOB_PREFIX = {NEW_SUITE: "sh_", CTRL_SUITE: "v3c_", MU_SUITE: "mu_"}

# Throttle by partition, not by job name: Stampede3's job limit is per user per queue (spr 24,
# skx 40) and const-vc-sh and const-vc-v3ctrl feed spr at the same time, so each feeder counts
# EVERY active job of ours in the batch script's partition.  The skx dd100 jobs no longer count.
COUNT_ACTIVE_NEW = r"""# Partition of the batch script: the throttle counts every active job of ours in it, whatever
# the name, because the Stampede3 limit is per user per queue (spr 24, skx 40) and the two 3.0
# suites (const-vc-sh, const-vc-v3ctrl) feed the same queue at once.  skx (dd100) jobs don't count.
PART="$(awk '/^#SBATCH[[:space:]]+-p[[:space:]]/ {print $3; exit}' "$SLURM_FILE")"
[[ -n "$PART" ]] || { echo "cannot read '#SBATCH -p' from $SLURM_FILE"; exit 1; }

count_active() {
  squeue -u "$USER" -h -p "$PART" -o "%t" \
  | awk '($1=="PD" || $1=="R" || $1=="CF" || $1=="CG") {c++} END {print c+0}'
}
"""

SUBMIT_ONE_NEW = r"""submit_one() {
  local id_num="$1"
  local runid jobname
  runid=$(printf 'run_%03d' "$id_num")                 # run directory / .prm name -- fixed
  jobname=$(printf '%s%03d' "$JOB_PREFIX" "$id_num")   # Slurm job name and log-file name only

  local prm="$BASE_DIR/$runid/$runid.prm"
  if [[ ! -f "$prm" ]]; then
    echo "SKIP: $prm not found"
    return 0
  fi

  local out="$SCRATCH/aspect_work/logs/${jobname}.%j.out"
  local err="$SCRATCH/aspect_work/logs/${jobname}.%j.err"

  # A rejected sbatch (e.g. the per-queue job limit, reachable when two feeders share the queue)
  # is retried after POLL seconds instead of dropping the run or killing the feeder.
  local sbout jid="" tries=0
  while :; do
    if sbout=$(sbatch -J "$jobname" -o "$out" -e "$err" \
                      --export=ALL,RUN_ID="$id_num",BASE_DIR="$BASE_DIR" \
                      "$SLURM_FILE" 2>&1); then
      jid=$(awk '/Submitted batch job/ {print $NF}' <<<"$sbout" | tail -n1)
      [[ -n "$jid" ]] && break
    fi
    tries=$((tries + 1))
    echo "WARN: sbatch did not accept $runid (try $tries): $sbout -- retrying in ${POLL}s"
    sleep "$POLL"
  done
  sleep 1
  echo "submitted $runid as job $jid ($jobname; active in $PART now: $(count_active))"
}
"""

# Per-suite scratch workspace.  $SCRATCH/aspect_work/ itself is the 2.5 suites' workspace
# (const-vc, ramped-vc, dd100 all cd there and write outputs/run_XXX), so a run_XXX of this
# suite launched there would, with 'Resume computation = auto', resume from -- or overwrite --
# the dd100 run of the same number, and the staged aspect.run_XXX / run_XXX.prm would collide
# too.  The .prm paths (outputs/run_XXX, inputs/) are relative, so a separate cwd is all it
# takes.  Pull with TACC_OUT=$SCRATCH/aspect_work/<suite>/outputs (pull_runs_from_tacc.sh
# defaults to that for these two suites).
SCRATCH_OLD = ('# --- scratch workspace ---\n'
               'RUN_SCRATCH="$SCRATCH/aspect_work/"\n')
SCRATCH_NEW = ('# --- scratch workspace (per suite) ---\n'
               "# NOT $SCRATCH/aspect_work/ itself: that is the 2.5 suites' workspace (const-vc, ramped-vc,\n"
               "# dd100 all write outputs/run_XXX there), and with 'Resume computation = auto' a run_XXX of\n"
               '# this suite would resume from -- or overwrite -- the dd100 run of the same number.\n'
               '# Outputs land in $RUN_SCRATCH/outputs/run_XXX; pull with TACC_OUT pointing there.\n'
               'RUN_SCRATCH="$SCRATCH/aspect_work/{suite}/"\n'
               'mkdir -p "$RUN_SCRATCH/inputs" "$RUN_SCRATCH/outputs"\n')

# Sapphire Rapids variant of run_one.slurm, written alongside it (run_one.spr.slurm).
# The skx-built ASPECT binary runs unchanged on spr -- SPR's instruction set is a superset
# of SKX's -- so only the queue and the rank count differ.  Both submitters honour
# SLURM_FILE, so the variant is selected with
#   SLURM_FILE=run_one.spr.slurm ./submit_from_list.sh <list>
SPR_NOTE = {
    NEW_SUITE: """#   * 112 ranks is a different domain decomposition from the 48-rank skx const-vc runs,
#     so the const-vc-sh/const-vc pair difference picks up decomposition noise on top of
#     the 2.5->3.x version change.  If production goes to spr, run const-vc-v3ctrl here
#     too (run_one.spr.slurm in that suite) so the control absorbs both.""",
    MU_SUITE: """#   * Same queue / rank count as const-vc-sh production, so the mu05-vs-sh pair
#     difference is the heating cap alone.""",
    CTRL_SUITE: """#   * This is the version-control set: it MUST run on the same queue/rank count as
#     const-vc-sh production, so that comparing it with the 2.5 skx/48-rank const-vc
#     outputs measures version + decomposition together.  If const-vc-sh goes to spr,
#     v3ctrl goes to spr.""",
}
SPR_HEADER = """
# Sapphire Rapids variant of run_one.slurm (which stays on skx, -n 48).
# Same binary: an SKX-built executable runs on SPR (SPR's ISA is a superset), so no
# separate compile is needed -- only the queue and the rank count change.
#
# Submit with:  SLURM_FILE=run_one.spr.slurm ./submit_from_list.sh pilot-list.txt
#
# Verified on the run_901 benchmark (run_089 to 0.5 Myr, 2026-09-22):
#   * 112 MPI processes = one 2x56-core Xeon Max node.
#   * Peak node RSS 45 GB at 0.5 Myr (batch-step MaxRSS; ibrun keeps all ranks in that
#     step) against 128 GB HBM.  A complete 48-rank 2.5 run peaks at 22 GB total, so the
#     AMR growth over 10.5 Myr cannot get near the limit.
#   * 6 min wall against ~11 min estimated for skx/48; spr is charged 2 SU/node-h (skx 1),
#     so about the same SUs per run at half the wall time.
#   * spr allows 24 jobs in queue per user (skx 40): the submitters' THROTTLE default is
#     24 for that reason -- an sbatch rejection is logged as WARN and NOT retried.
{note}"""

# the template must carry exactly the same edits, so the record stays in step
tmpl = TEMPLATE.read_text()
for frag in (NEW_FORM, NEW_VIS, NEW_AMG):
    assert tmpl.count(frag) == 1, f"template out of step with this script: {frag[:40]!r}"

if not SRC_DIR.is_dir():
    sys.exit(f"const-vc run-inputs not found at {SRC_DIR}; regenerate them first "
             "(build_runs.const-vc.py) or adapt this script to the template.")
OUT_DIR.mkdir(parents=True, exist_ok=True)

n_extra_lines = ((NEW_FORM.count("\n") - OLD_FORM.count("\n"))
                 + (NEW_VIS.count("\n") - OLD_VIS.count("\n")))
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
        suite = new_base.rsplit("/", 1)[-1]
        if sname.startswith("submit_"):
            # throttle 24 = spr per-user queue limit; counted per partition (COUNT_ACTIVE_NEW)
            assert t.count('THROTTLE="${THROTTLE:-30}"') == 1, f"{sname}: THROTTLE line not found"
            t = re.sub(r'THROTTLE="\$\{THROTTLE:-30\}".*\n',
                       'THROTTLE="${THROTTLE:-24}"  # max active jobs in the partition; spr per-user limit is 24 (skx 40)\n', t)
            assert t.count('JOB_PREFIX="${JOB_PREFIX:-run_}"') == 1, f"{sname}: JOB_PREFIX line not found"
            t = re.sub(r'JOB_PREFIX="\$\{JOB_PREFIX:-run_\}".*\n',
                       f'JOB_PREFIX="${{JOB_PREFIX:-{JOB_PREFIX[suite]}}}"  # Slurm job/log name only; run dirs stay run_XXX\n', t)
            t, n = re.subn(r'count_active\(\) \{.*?\n\}\n', COUNT_ACTIVE_NEW, t, count=1, flags=re.S)
            assert n == 1, f"{sname}: count_active not found"
            t, n = re.subn(r'submit_one\(\) \{.*?\n\}\n', SUBMIT_ONE_NEW, t, count=1, flags=re.S)
            assert n == 1, f"{sname}: submit_one not found"
            t = t.replace("(counts PD+R)", "(active jobs in partition $PART, any name)")
        if sname == "run_one.slurm":
            # Own workspace -- see SCRATCH_NEW.  Logs stay shared (job id in the name).
            assert t.count(SCRATCH_OLD) == 1, "run_one.slurm: RUN_SCRATCH line not found"
            t = t.replace(SCRATCH_OLD, SCRATCH_NEW.format(suite=suite))
            t = t.replace('ASPECT_EXE="${asp25_skx:?asp25_skx not set}"',
                          '# TODO(const-vc-sh): this suite needs an ASPECT >= 3.0 binary (stress-limited shear heating);\n'
                          '# swap the module line above and the variable below for the 3.x build before submitting.\n'
                          'ASPECT_EXE="${asp3_skx:?asp3_skx not set (ASPECT 3.x build, see const-vc-sh/README.md)}"')
        (dst_dir / sname).write_text(t)
        shutil.copymode(src, dst_dir / sname)
        if sname == "run_one.slurm":
            spr = t.replace("#SBATCH -p skx", "#SBATCH -p spr")
            spr = spr.replace("#SBATCH -n 48\n",
                              "#SBATCH -n 112\n" + SPR_HEADER.format(note=SPR_NOTE[suite]) + "\n")
            spr = spr.replace(
                "# TODO(const-vc-sh): this suite needs an ASPECT >= 3.0 binary (stress-limited shear heating);\n"
                "# swap the module line above and the variable below for the 3.x build before submitting.",
                f"# TODO({suite}): this suite needs an ASPECT >= 3.0 binary (stress-limited shear heating).\n"
                "# If the 3.x build links a different deal.II than the 9.5 enable.sh above, swap that line too.")
            (dst_dir / "run_one.spr.slurm").write_text(spr)
            shutil.copymode(src, dst_dir / "run_one.spr.slurm")

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
        assert text.count(OLD_VIS) == 1
        new = text.replace(OLD_AMG, NEW_AMG).replace(OLD_VIS, CTRL_VIS).replace(
            "# This file is auto-generated by build_runs.const-vc.py",
            "# This file is auto-generated by build_runs.const-vc-sh.py --control (target: ASPECT >= 3.0)\n"
            f"# VERSION CONTROL: {SRC_SUITE}/run-inputs/{name}/{name}.prm with only 'Stokes solver type = block AMG'\n"
            "# made explicit (3.0 defaults to GMG) and the visualization output switched to the 3.0 spelling\n"
            "# ('viscosity' -> 'material properties'; same vtu field).  No heating.  Compare with the\n"
            "# existing const-vc (2.5) output.")
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

OLD_CAP = ("    set Cohesion for maximum shear stress = 10e6\n"
           "    set Friction angle for maximum shear stress = 0.523599\n")
NEW_CAP = ("    set Cohesion for maximum shear stress = 1e6        # const-vc-sh-mu05: 1 MPa (was 10 MPa)\n"
           "    set Friction angle for maximum shear stress = 0.050021   # asin(0.05): mu' = 0.05 (was 0.523599 = 30 deg)\n")
n_mu = 0
if MU05:
    pilot = [l.strip().zfill(3) for l in PILOT_LIST.read_text().split() if l.strip()]
    MU_DIR.mkdir(parents=True, exist_ok=True)
    for rid in pilot:
        name = f"run_{rid}"
        text = (OUT_DIR / name / f"{name}.prm").read_text()      # derived from the const-vc-sh .prm
        assert text.count(OLD_CAP) == 1, f"{name}: heating cap lines not found once"
        new = text.replace(OLD_CAP, NEW_CAP).replace(
            "# This file is auto-generated by build_runs.const-vc-sh.py (target: ASPECT >= 3.0)",
            "# This file is auto-generated by build_runs.const-vc-sh.py --mu05 (target: ASPECT >= 3.0)\n"
            f"# LOW-FRICTION HEATING CAP PILOT: {NEW_SUITE}/run-inputs/{name}/{name}.prm with only the\n"
            "# shear-heating stress limiter changed: cohesion 1 MPa, friction angle asin(0.05) = 0.050021 rad,\n"
            "# i.e. heating stress = min(2 eta eps, 1 MPa + 0.05 P) (Kohn et al. 2018 mu' = 0.05).\n"
            "# Flow/rheology identical to const-vc-sh and const-vc: the cap enters the heating term only.")
        assert len(new.splitlines()) == len(text.splitlines()) + 4
        dst_run = MU_DIR / name
        (dst_run / "inputs").mkdir(parents=True, exist_ok=True)
        (dst_run / f"{name}.prm").write_text(new)
        for f in sorted((OUT_DIR / name / "inputs").iterdir()):
            dst = dst_run / "inputs" / f.name
            if not dst.exists():
                try:
                    os.link(f, dst)
                except OSError:
                    shutil.copy2(f, dst)
        n_mu += 1
    copy_scripts(MU_DIR, f"production-runs_v2/{MU_SUITE}")
    shutil.copy2(PILOT_LIST, MU_DIR / "pilot-list.txt")

print(f"Finished: {n_ok} run directories under {OUT_DIR} "
      f"(prm: Formulation custom + stress-limited shear heating + block AMG explicit + 'heating' output; "
      f"inputs hard-linked from {SRC_SUITE}); scripts: {', '.join(SCRIPTS)} + run_one.spr.slurm "
      f"with BASE_DIR -> {NEW_BASE}"
      + (f"; version-control set: {n_ctrl} runs under {CTRL_DIR}" if CONTROL else "")
      + (f"; mu'=0.05 heating-cap pilot: {n_mu} runs under {MU_DIR}" if MU05 else ""))
