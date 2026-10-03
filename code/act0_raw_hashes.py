"""Act-0 re-audit (2026-10-03): size + SHA-256 of every raw recording file behind the report, so a reader can check their
copy byte for byte without us redistributing any EEG. TUAB = the 2,993 EDFs under data/TUAB/edf (unofficial mirror,
see report §5.5.3); NMT = every file under data/NMT-Scalp-EEG; stroke = every file under data/zenodo_stroke.

  python act0_raw_hashes.py   ->  docs/act0_repro/raw_recordings_sha256.csv  (dataset, relative_path, size_bytes, sha256)
"""
import csv
import glob
import hashlib
import os

from hw_guard import cool_gate

ROOT = "/home/mashfiq/eeg_vjepa"
SETS = {"TUAB": f"{ROOT}/data/TUAB/edf/**/*.edf", "NMT": f"{ROOT}/data/NMT-Scalp-EEG/**/*",
        "stroke": f"{ROOT}/data/zenodo_stroke/**/*"}
OUT = f"{ROOT}/docs/act0_repro/raw_recordings_sha256.csv"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


if __name__ == "__main__":
    rows = []
    for name, pat in SETS.items():
        base = pat.split("/**")[0]
        fl = sorted(p for p in glob.glob(pat, recursive=True) if os.path.isfile(p))
        for i, p in enumerate(fl):
            if i % 10 == 0:
                cool_gate(pause=80.0, resume=70.0, abort=92.0, verbose=False)
            rows.append((name, os.path.relpath(p, base), os.path.getsize(p), sha256(p)))
        print(f"{name}: {len(fl)} files", flush=True)
    tmp = OUT + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f); w.writerow(("dataset", "relative_path", "size_bytes", "sha256")); w.writerows(rows)
    os.replace(tmp, OUT)
    assert sum(r[0] == "TUAB" for r in rows) == 2993, "expected the 2,993 TUAB EDFs"
