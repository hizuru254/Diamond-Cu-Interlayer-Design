#!/usr/bin/env python3
"""Piecewise fit for diffuse interfaces: three bulk segments plus two jumps.

A sharp-interface extrapolation fails when the interlayer and its interfaces mix
over 10+ A.  Instead we fit the whole profile with one continuous piecewise
linear model

    T(z) = a + b1 (z - z1)                            for z < z1
    T(z) = a + J1 + b2 (z - z1)                       for z1 <= z < z2
    T(z) = a + J1 + b2 (z2 - z1) + J2 + b3 (z - z2)   for z >= z2

where J1, J2 are the two interface temperature jumps and b1,b2,b3 are the bulk
gradients.  All six parameters are linear, so a weighted least-squares solve
gives everything at once; weights are the atom counts per bin.

Outputs
    R_total = (J1 + b2*(z2-z1) + J2) / q      (should match tbr_stack)
    R_i     = J_i / q                          (per-interface, even when diffuse)
    kappa_i = q / (-b_i)                       (local bulk conductivity)
"""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import geomlib
from tbr_plane import parse_blocks


def design(z, z1, z2):
    Z = np.asarray(z, dtype=float)
    n = len(Z)
    A = np.zeros((n, 6))
    r1 = Z < z1
    r2 = (Z >= z1) & (Z < z2)
    r3 = Z >= z2
    A[r1, 0] = 1.0; A[r1, 1] = Z[r1] - z1
    A[r2, 0] = 1.0; A[r2, 2] = 1.0; A[r2, 3] = Z[r2] - z1
    A[r3, 0] = 1.0; A[r3, 2] = 1.0
    A[r3, 3] = z2 - z1; A[r3, 4] = 1.0; A[r3, 5] = Z[r3] - z2
    return A, r1, r2, r3


def fit_block(blocks, z1, z2):
    z, T, w = profile_counts(blocks)
    m = np.isfinite(T) & (w > 0)
    A, r1, r2, r3 = design(z[m], z1, z2)
    if r1.sum() < 3 or r2.sum() < 2 or r3.sum() < 3:
        return None
    W = np.sqrt(w[m])[:, None]
    co, *_ = np.linalg.lstsq(A * W, T[m] * np.sqrt(w[m]), rcond=None)
    a, b1, J1, b2, J2, b3 = co
    pred = A @ co
    ss = (w[m] * (T[m] - pred) ** 2).sum()
    ss_tot = (w[m] * (T[m] - np.average(T[m], weights=w[m])) ** 2).sum()
    r2score = 1 - ss / ss_tot if ss_tot > 0 else float("nan")
    return dict(a=a, b1=b1, J1=J1, b2=b2, J2=J2, b3=b3, r2=r2score,
                n1=int(r1.sum()), n2=int(r2.sum()), n3=int(r3.sum()))


def profile_counts(blocks):
    """Return (z, T, weight) with weight = total atom count per bin."""
    nb = len(blocks[0]["rows"])
    acc = np.zeros(nb); cnt = np.zeros(nb, int); wsum = np.zeros(nb)
    z = np.array([r[1] for r in blocks[0]["rows"]])
    for b in blocks:
        a = np.array(b["rows"])
        occ = a[:, 2] > 0
        acc[occ] += a[occ, 3]; cnt[occ] += 1
        wsum += a[:, 2]
    T = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
    return z, T, wsum / max(len(blocks), 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--seed", default="seed_1001")
    ap.add_argument("--blocks", type=int, default=10)
    a = ap.parse_args()
    run = Path(a.run_dir).resolve()
    geom = geomlib.load_json(run/"geom.json")
    params = geomlib.load_json(run/"params.snapshot.json")
    zs = sorted(float(i["z"]) for i in geom["interfaces"])
    if len(zs) != 2:
        print("[SKIP] needs two interfaces"); return 1
    z1, z2 = zs
    area = float(geom["box"]["lx"]) * float(geom["box"]["ly"])
    q = (float(params["flux_ev_per_ps"]) * float(params["units"]["ev_per_ps_to_W"])
         / (area * float(params["units"]["ang2_to_m2"])))

    blocks = parse_blocks(run/a.seed/"profile.txt")
    f = fit_block(blocks, z1, z2)
    if f is None:
        print("[FAIL] not enough bins in one region"); return 1
    print(f"{run.name}: piecewise fit  R2 = {f['r2']:.4f}   "
          f"bins {f['n1']}/{f['n2']}/{f['n3']}")
    R1, R2 = f["J1"]/q, f["J2"]/q
    Rt = (f["J1"] - f["b2"]*(z2-z1) + f["J2"])/q
    print(f"   J1 = {f['J1']:7.2f} K   J2 = {f['J2']:7.2f} K")
    print(f"   R(DIA/INT) = {R1:.3e}   R_int_bulk = {-f['b2']*(z2-z1)/q:.3e}   "
          f"R(INT/CU) = {R2:.3e}   R_total = {Rt:.3e} m2K/W")
    for nm, b in (("diamond", f["b1"]), ("interlayer", f["b2"]), ("Cu", f["b3"])):
        if b < 0:
            print(f"   kappa({nm}) = {q/(-b*1e10):.3f} W/mK")
        else:
            print(f"   kappa({nm}) = negative gradient ({b:.2e} K/A) - suspicious")

    # block error bars
    idx = np.linspace(0, len(blocks), a.blocks+1).astype(int)
    tot, r1s, r2s = [], [], []
    for i, j in zip(idx[:-1], idx[1:]):
        if j <= i: continue
        ff = fit_block(blocks[i:j], z1, z2)
        if not ff: continue
        tot.append((ff["J1"] - ff["b2"]*(z2-z1) + ff["J2"])/q)
        r1s.append(ff["J1"]/q); r2s.append(ff["J2"]/q)
    for nm, arr in (("R_total", tot), ("R_DIA_INT", r1s), ("R_INT_CU", r2s)):
        arr = np.array(arr)
        if len(arr) > 2:
            print(f"   {nm:10s} block mean {arr.mean():.3e} +/- {arr.std(ddof=1)/math.sqrt(len(arr)):.1e}")

    geomlib.dump_json(run/a.seed/"tbr_piecewise.json",
                      {"R_total": Rt, "R_DIA_INT": R1, "R_INT_CU": R2,
                       "R_bulk_INT": -f["b2"]*(z2-z1)/q,
                       "J1": f["J1"], "J2": f["J2"], "r2": f["r2"], "q": q})
    return 0


if __name__ == "__main__":
    sys.exit(main())
