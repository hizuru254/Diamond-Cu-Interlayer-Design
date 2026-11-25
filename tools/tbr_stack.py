#!/usr/bin/env python3
"""Stack-level thermal resistance for a two-interface cell (diamond/INT/Cu).

Two complementary numbers, both from the same temperature profile:

  R_stack  = [T(z1 - d) - T(z2 + d)] / q      d = 15 A fixed
             the resistance the device actually sees across the interlayer; it
             contains R(DIA|INT) + R_bulk(INT) + R(INT|CU) plus 15 A of diamond
             bulk and 15 A of Cu bulk, which are identical for every design and
             therefore cancel in comparisons.

  R_iface  = dT between z_i - d and z_i + d, extrapolated to d -> 0 with a
             linear fit over d = 4..12 A, for each interface separately.

Errors come from block averaging of the production segment.
"""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import geomlib
from tbr_plane import parse_blocks, profile


def T_at(z, T, zp):
    m = np.isfinite(T)
    zz, tt = z[m], T[m]
    if zp < zz.min() or zp > zz.max():
        raise ValueError(f"z={zp} outside {zz.min():.1f}..{zz.max():.1f}")
    return float(np.interp(zp, zz, tt))


def q_of(run: Path) -> float:
    geom = geomlib.load_json(run/"geom.json")
    params = geomlib.load_json(run/"params.snapshot.json")
    area = float(geom["box"]["lx"]) * float(geom["box"]["ly"])
    return (float(params["flux_ev_per_ps"]) * float(params["units"]["ev_per_ps_to_W"])
            / (area * float(params["units"]["ang2_to_m2"])))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--seed", default="seed_1001")
    ap.add_argument("--d", type=float, default=15.0,
                    help="distance (A) from each interface to its measuring plane")
    ap.add_argument("--blocks", type=int, default=10)
    a = ap.parse_args()
    run = Path(a.run_dir).resolve()
    sd = run/a.seed
    geom = geomlib.load_json(run/"geom.json")
    zs = [float(i["z"]) for i in geom["interfaces"]]
    if len(zs) != 2:
        print(f"[SKIP] needs exactly two interfaces, found {len(zs)}"); return 1
    z1, z2 = sorted(zs)
    q = q_of(run)
    blocks = parse_blocks(sd/"profile.txt")

    def stack_R(blocks_sub):
        z, T = profile(blocks_sub)
        return (T_at(z, T, z1-a.d) - T_at(z, T, z2+a.d)) / q

    R = stack_R(blocks)
    idx = np.linspace(0, len(blocks), a.blocks+1).astype(int)
    per = []
    for i, j in zip(idx[:-1], idx[1:]):
        if j <= i: continue
        per.append(stack_R(blocks[i:j]))
    per = np.array(per)
    sem = per.std(ddof=1)/math.sqrt(len(per)) if len(per) > 1 else float("nan")
    half = len(per)//2
    rel = abs(per[:half].mean()-per[half:].mean())/per[half:].mean() if half else float("nan")

    print(f"{run.name}: R_stack(z1-{a.d:.0f} -> z2+{a.d:.0f}) = {R:.4e} m2K/W "
          f"+/- {sem:.1e} ({100*sem/R:.1f}%)  half-to-half {100*rel:.1f}%")

    # per-interface extrapolation
    for name, zc in zip(("DIA_INT", "INT_CU"), (z1, z2)):
        ds, vals = [], []
        for d in (4, 6, 8, 10, 12):
            try:
                z, T = profile(blocks)
                ds.append(d); vals.append((T_at(z, T, zc-d) - T_at(z, T, zc+d))/q)
            except Exception:
                pass
        if len(ds) >= 3:
            co = np.polyfit(ds, vals, 1)
            r2 = 1 - ((np.array(vals)-np.polyval(co, ds))**2).sum() / \
                     ((np.array(vals)-np.mean(vals))**2).sum()
            print(f"   {name:8s} R(d) = {co[0]:.2e}*d + {co[1]:.3e}   R_int = {co[1]:.3e}  R2={r2:.3f}")
        else:
            print(f"   {name:8s} not enough valid planes (thin interlayer)")

    out = {"run": run.name, "R_stack": R, "R_stack_sem": sem,
           "q_W_m2": q, "z_interfaces": [z1, z2], "half_to_half": rel}
    geomlib.dump_json(sd/"tbr_stack.json", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
