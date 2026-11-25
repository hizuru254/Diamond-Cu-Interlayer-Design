#!/usr/bin/env python3
"""Retest WC (2-atom P-6m2 cell) and MoC with a wider scan."""
import sys, re, subprocess
from pathlib import Path
import numpy as np
sys.path.insert(0, "/home/cc/interlayer_nemd/tools")
import capability_matrix as cm

ROOT = Path("/home/cc/interlayer_nemd")
W = Path("/tmp/cap2"); W.mkdir(parents=True, exist_ok=True)
cm.W = W

def wc_energy(a, cr):
    """WC P-6m2: W at (0,0,0); C at (1/3,2/3,1/2)."""
    c = a*cr; n = 3; nz = 3
    cy = a*np.sqrt(3)/2
    pts, types = [], []
    for i in range(n):
        for j in range(n):
            for k in range(nz):
                for fx, fy, fz, t in ((0,0,0,1), (1/3,2/3,.5,2)):
                    pts.append(((i+fx)*a+(j+fy)*(-a/2), (j+fy)*cy, (k+fz)*c))
                    types.append(t)
    m = [183.84, 12.011]
    L = ["WC", "", f"{len(pts)} atoms", "2 atom types", "",
         f"0 {n*a:.6f} xlo xhi", f"0 {n*cy:.6f} ylo yhi", f"0 {nz*c:.6f} zlo zhi",
         f"{-a/2:.6f} 0.0 0.0 xy xz yz", "", "Masses", ""]
    for i, mm in enumerate(m, 1): L.append(f"{i} {mm}")
    L += ["", "Atoms # atomic", ""]
    for i, (p, t) in enumerate(zip(pts, types), 1):
        L.append(f"{i} {t} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}")
    (W/"c.data").write_text("\n".join(L)+"\n", encoding="utf-8")
    inp = f"""units metal
atom_style atomic
read_data c.data
pair_style nep
pair_coeff * * {cm.MODEL} W C
neighbor 2.0 bin
neigh_modify one 4000
thermo_style custom step pe
minimize 1e-12 1e-12 5000 50000
print "EPER $(pe)"
"""
    (W/"in.e").write_text(inp, encoding="utf-8")
    subprocess.run([cm.EXE,"-in","in.e","-log","log.e","-screen","none"], cwd=W,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t = (W/"log.e").read_text(errors="replace")
    m2 = re.findall(r"EPER\s+(-?\d+\.\d+)", t)
    return (float(m2[-1])/len(pts) if m2 else None)

def scan(fn, aexp, cr, lo=0.90, hi=1.12, step=0.01):
    xs, ys = [], []
    for f in np.arange(lo, hi+1e-9, step):
        e = fn(aexp*f, cr)
        if e is not None: xs.append(aexp*f); ys.append(e)
    xs, ys = np.array(xs), np.array(ys)
    co = np.polyfit(xs, ys, 2)
    v = -co[1]/(2*co[0])
    return float(v), float(np.polyval(co, v)), bool(co[0] > 0), bool(xs.min() < v < xs.max())

print("WC (2-atom P-6m2, c/a=0.976):")
a0, e0, conv, ins = scan(wc_energy, 2.906, 0.976)
print(f"  a0={a0:.3f} (exp 2.906, {100*(a0-2.906)/2.906:+.2f}%) E0={e0:.3f} convex={conv} inside={ins}")

print("MoC rocksalt wider scan (0.85-1.20):")
a0, e0, conv, ins = scan(lambda a, cr: cm.energy("rocksalt", ("Mo","C"), a, None)[0],
                         4.28, None, lo=0.85, hi=1.20)
print(f"  a0={a0:.3f} (exp 4.28, {100*(a0-4.28)/4.28:+.2f}%) E0={e0:.3f} convex={conv} inside={ins}")
