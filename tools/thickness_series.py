#!/usr/bin/env python3
"""Separate interface resistance from interlayer bulk conductivity.

For a series of runs with interlayer thickness t_i we have

    R_stack(t) = R_interfaces + t / kappa_interlayer

so a linear fit of R_stack against t gives the intercept (the two interfaces
together) and the slope (1/kappa).  This is the whole point of a thickness
series: it turns a single number into two physical quantities.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", nargs="+", required=True, help="thickness_A:run_dir")
    ap.add_argument("--out", default="reports/thickness_series.json")
    a = ap.parse_args()
    ts, rs = [], []
    for p in a.points:
        t, rd = p.split(":")
        f = Path(ROOT/rd)/"seed_1001"/"tbr_stack.json"
        if not f.exists():
            print(f"[skip] {rd}: no tbr_stack.json"); continue
        d = json.loads(f.read_text())
        ts.append(float(t)); rs.append(d["R_stack"])
        print(f"  t = {float(t):5.1f} A   R_stack = {d['R_stack']:.4e}")
    if len(ts) < 2:
        print("[FAIL] need at least two thicknesses"); return 1
    ts = np.array(ts); rs = np.array(rs)
    co = np.polyfit(ts, rs, 1)
    r2 = 1 - ((rs - np.polyval(co, ts))**2).sum() / ((rs - rs.mean())**2).sum()
    kappa = 1.0/co[0]/1e10 if co[0] > 0 else float("nan")   # A -> m
    print(f"\n[R(t)] = {co[0]:.3e} * t + {co[1]:.3e}   R2 = {r2:.4f}")
    print(f"  interface part (t->0)  R_if = {co[1]:.3e} m2K/W")
    print(f"  interlayer kappa       = {kappa:.2f} W/mK")
    (ROOT/a.out).write_text(json.dumps({"t_A": ts.tolist(), "R_stack": rs.tolist(),
                                        "slope": co[0], "intercept": co[1],
                                        "r2": r2, "kappa_W_mK": kappa}, indent=2),
                            encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
