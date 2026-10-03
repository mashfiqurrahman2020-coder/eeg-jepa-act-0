"""
Thermal guard for local experiments on this workstation (it hard-resets when the
CPU overheats). Reads coretemp/GPU temps straight from sysfs (no load) and provides
a cool_gate() to pause heavy loops until the CPU is back in a safe range.

Usage in a heavy loop:
    from hw_guard import cool_gate, temps, cap_threads
    cap_threads(4)                      # limit BLAS/torch threads before importing heavy libs
    for chunk in work:
        cool_gate()                     # blocks if too hot, aborts if it can't cool
        ... do a bounded chunk of work ...

Defaults are conservative because idle package temp is ~58C and the box reset in the
90s C: pause at 85C, resume at 70C, abort if still >92C after the cooldown budget.
"""
import glob, os, time

PAUSE_C = 88.0        # stop feeding work before the 92C safety ceiling
RESUME_C = 86.0       # resume once cooled to at/below this
ABORT_C = 92.0        # hard-stop at the 92C safety ceiling
COOLDOWN_BUDGET = 180 # seconds max to wait for cooldown before aborting


def _read_coretemp_c():
    """Max coretemp (Package/cores) in C from sysfs, or None if unavailable."""
    best = None
    for hw in glob.glob("/sys/class/hwmon/hwmon*"):
        try:
            name = open(os.path.join(hw, "name")).read().strip()
        except OSError:
            continue
        if name not in ("coretemp", "k10temp", "zenpower"):
            continue
        for tf in glob.glob(os.path.join(hw, "temp*_input")):
            try:
                v = int(open(tf).read().strip()) / 1000.0
            except (OSError, ValueError):
                continue
            best = v if best is None else max(best, v)
    return best


def _read_gpu_c():
    try:
        import subprocess
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5)
        return float(out.stdout.strip().split("\n")[0])
    except Exception:
        return None


def temps():
    return {"cpu_C": _read_coretemp_c(), "gpu_C": _read_gpu_c()}


def cap_threads(n=4):
    """Cap BLAS/OpenMP threads. Call BEFORE importing numpy/torch for it to take effect.
    HW_THREADS env overrides n (to run a side job gentler without editing it)."""
    n = int(os.environ.get("HW_THREADS", n))
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = str(n)


def cool_gate(pause=PAUSE_C, resume=RESUME_C, abort=ABORT_C, budget=COOLDOWN_BUDGET,
              verbose=False):   # quiet by default: no routine pause/resume log spam (abort still raises)
    """If CPU >= pause, block (polling) until <= resume. Raise RuntimeError if it can't
    get below `abort` within `budget` seconds. No-op if temp can't be read."""
    t = _read_coretemp_c()
    if t is None or t < pause:
        return t
    if verbose:
        print(f"[hw_guard] CPU {t:.0f}C >= {pause:.0f}C — pausing to cool...", flush=True)
    waited = 0.0
    while True:
        time.sleep(5.0)
        waited += 5.0
        t = _read_coretemp_c()
        if t is None or t <= resume:
            if verbose:
                print(f"[hw_guard] cooled to {t:.0f}C — resuming.", flush=True)
            return t
        if waited >= budget and t > abort:
            raise RuntimeError(
                f"[hw_guard] CPU still {t:.0f}C after {budget}s cooldown — aborting to "
                f"protect hardware.")


if __name__ == "__main__":
    print("current temps:", temps())
    print(f"thresholds: pause>={PAUSE_C} resume<={RESUME_C} abort>{ABORT_C}")
    cool_gate()   # exercises the reader; no-op when cool
    print("cool_gate OK (system within safe range)")
