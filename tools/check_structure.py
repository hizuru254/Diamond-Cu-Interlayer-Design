#!/usr/bin/env python3
"""Structure integrity check: is every species still in its own layer?

This is the check that caught the first failed run (Cu sank, diamond rose). It is
deliberately independent of the analysis tools: it reads final.data, assigns each
atom to the layer whose z-range contains it, and reports how many atoms of each
species sit outside their own layer (interdiffusion) and whether the layer order
in z is still diamond / interlayer / Cu.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--geom", required=True)
    a = ap.parse_args()
    geom = json.loads(Path(a.geom).read_text(encoding="utf-8"))
    layers = geom["layers"]
    d = Path(a.data).read_text().splitlines()
    ia = next(i for i, l in enumerate(d) if l.strip().startswith("Atoms"))
    ty, z = [], []
    for l in d[ia+2:]:
        s = l.split()
        if len(s) < 5 or not s[0].isdigit():
            break
        ty.append(int(s[1])); z.append(float(s[4]))
    ty = np.array(ty); z = np.array(z)
    m = np.isfinite(z)
    print(f"atoms {len(ty)}  z {z.min():.1f}..{z.max():.1f}")
    for L in layers:
        sel = (z >= L["z_min"]) & (z <= L["z_max"])
        c = np.bincount(ty[sel], minlength=4)[1:]
        print(f"  {L['id']:4s} z {L['z_min']:7.2f}..{L['z_max']:7.2f}  n={sel.sum():5d}  C/Ti/Cu={c}")
    # order check: mean z of the diamond-type atoms vs the Cu-type atoms
    zc = z[ty == 1]; zcu = z[ty == 3]
    ok = True
    if len(zc) and len(zcu):
        print(f"  mean z: C {zc.mean():.1f}   Cu {zcu.mean():.1f}")
        if zc.mean() > zcu.mean():
            print("  [FAIL] diamond is ABOVE Cu - the stack inverted"); ok = False
    # interdiffusion: non-C atoms inside the diamond layer, C inside the Cu layer
    dia = layers[0]; cu = layers[-1]
    in_dia = (z >= dia["z_min"]) & (z <= dia["z_max"])
    in_cu = (z >= cu["z_min"]) & (z <= cu["z_max"])
    n_non_c_in_dia = int(((ty != 1) & in_dia).sum())
    n_c_in_cu = int(((ty == 1) & in_cu).sum())
    print(f"  interdiffusion: {n_non_c_in_dia} non-C in the diamond layer, "
          f"{n_c_in_cu} C in the Cu layer")
    if not ok:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
