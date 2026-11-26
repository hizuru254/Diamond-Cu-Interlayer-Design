#!/usr/bin/env python3
"""Acceptance gate for one run. Five hard criteria, printed as PASS/FAIL.

1 structure   no interdiffusion, layer order correct
2 profile     piecewise fit R2 > 0.95 and the profile decreases with z
3 consistency |R_piecewise - R_stack| / R_stack < 0.15
4 stationarity half-to-half scatter < 10 %
5 stability   |dPE/dt| < 0.01 eV/ps over the production half
"""
from __future__ import annotations
import argparse, json, re, subprocess, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def run(cmd):
    r = subprocess.run(cmd, shell=True, cwd=ROOT, capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    a = ap.parse_args()
    rd = Path(a.run_dir)
    sd = rd/"seed_1001"
    ok = {}

    # 1 structure
    rc, out = run(f"{PY} tools/check_structure.py --data {sd}/final.data --geom {rd}/geom.json")
    inter = re.search(r"interdiffusion: (\d+) non-C.*?, (\d+) C in", out)
    n1, n2 = (int(inter.group(1)), int(inter.group(2))) if inter else (999, 999)
    ok["structure"] = (rc == 0 and n1 == 0 and n2 == 0)
    print(f"[1] structure      {'PASS' if ok['structure'] else 'FAIL'}  "
          f"(interdiffusion {n1}/{n2})")

    # 2 profile
    rc, out = run(f"{PY} tools/tbr_piecewise.py --run-dir {rd} --seed seed_1001")
    m = re.search(r"R2 = ([\d.]+)", out)
    r2 = float(m.group(1)) if m else 0.0
    ok["profile"] = (rc == 0 and r2 > 0.95)
    print(f"[2] profile        {'PASS' if ok['profile'] else 'FAIL'}  (R2 = {r2:.3f})")

    # 3 consistency
    rc2, out2 = run(f"{PY} tools/tbr_stack.py --run-dir {rd} --seed seed_1001")
    ms = re.search(r"R_stack.*?= ([\d.eE+-]+)", out2)
    mp = re.search(r"R_total = ([\d.eE+-]+)", out)
    R_stack = float(ms.group(1)) if ms else float("nan")
    R_pw = float(mp.group(1)) if mp else float("nan")
    rel = abs(R_pw - R_stack)/abs(R_stack) if np.isfinite(R_stack) and R_stack else 9.9
    ok["consistency"] = rel < 0.15
    print(f"[3] consistency    {'PASS' if ok['consistency'] else 'FAIL'}  "
          f"(R_piecewise {R_pw:.3e} vs R_stack {R_stack:.3e}, {100*rel:.1f}%)")

    # 4 stationarity
    mh = re.search(r"half-to-half ([\d.-]+)%", out2)
    hh = abs(float(mh.group(1))) if mh else 999
    ok["stationarity"] = hh < 10.0
    print(f"[4] stationarity   {'PASS' if ok['stationarity'] else 'FAIL'}  (half-to-half {hh:.1f}%)")

    # 5 stability
    rc5, out5 = run(f"{PY} tools/verify_run.py {rd} seed_1001")
    md = re.search(r"dPE/dt = ([\d.eE+-]+)", out5)
    dpe = abs(float(md.group(1))) if md else 999
    ok["stability"] = dpe < 0.01
    print(f"[5] stability      {'PASS' if ok['stability'] else 'FAIL'}  (|dPE/dt| {dpe:.4f} eV/ps)")

    allok = all(ok.values())
    print(f"\nVERDICT: {'ACCEPT' if allok else 'REJECT'}   "
          f"R_stack = {R_stack:.3e} m2K/W")
    (ROOT/f"reports/accept_{rd.name}.json").write_text(
        json.dumps({"run": rd.name, "criteria": ok, "R_stack": R_stack,
                    "R_piecewise": R_pw, "r2": r2, "half_to_half": hh,
                    "dpe_dt": dpe, "verdict": "ACCEPT" if allok else "REJECT"}, indent=2),
        encoding="utf-8")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
