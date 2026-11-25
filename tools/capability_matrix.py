#!/usr/bin/env python3
"""Capability matrix: can the Cu/Ti/C NEP describe bulk carbides and metals?

For every candidate phase we scan the lattice parameter, minimise, and report
  a0, E0/atom and whether E(a) is convex (i.e. a real minimum exists).
A material that fails this test cannot enter the design space.
"""
from __future__ import annotations
import re, subprocess, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EXE = "/home/cc/lammps/src/lmp_mpi"
MODEL = str(ROOT / "potentials/CuTiC_NEP.txt")
W = Path("/tmp/cap"); W.mkdir(parents=True, exist_ok=True)

MASS = {"C": 12.011, "Cu": 63.546, "Ti": 47.867, "Cr": 52.00, "Mo": 95.95, "W": 183.84}

def cell(kind, els, a, covera=None, n=3, nz=3):
    """Return (positions, types, box). els is a tuple of element names per type."""
    if kind == "fcc":
        basis = [(0,0,0),(0,.5,.5),(.5,0,.5),(.5,.5,0)]; L = a*n
        pts = [((i+bx)*a,(j+by)*a,(k+bz)*a) for i in range(n) for j in range(n)
               for k in range(n) for bx,by,bz in basis]
        types = [1]*len(pts); box = (L,L,L); tilt = (0,0,0)
    elif kind == "bcc":
        basis = [(0,0,0),(.5,.5,.5)]; L = a*n
        pts = [((i+bx)*a,(j+by)*a,(k+bz)*a) for i in range(n) for j in range(n)
               for k in range(n) for bx,by,bz in basis]
        types = [1]*len(pts); box = (L,L,L); tilt = (0,0,0)
    elif kind == "diamond":
        basis = [(0,0,0),(0,.5,.5),(.5,0,.5),(.5,.5,0),
                 (.25,.25,.25),(.25,.75,.75),(.75,.25,.75),(.75,.75,.25)]
        L = a*n
        pts = [((i+bx)*a,(j+by)*a,(k+bz)*a) for i in range(n) for j in range(n)
               for k in range(n) for bx,by,bz in basis]
        types = [1]*len(pts); box = (L,L,L); tilt = (0,0,0)
    elif kind == "rocksalt":
        basis = [(0,0,0),(.5,.5,0),(.5,0,.5),(0,.5,.5),
                 (.5,.5,.5),(0,0,.5),(0,.5,0),(.5,0,0)]
        L = a*n
        pts, types = [], []
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for m,(bx,by,bz) in enumerate(basis):
                        pts.append(((i+bx)*a,(j+by)*a,(k+bz)*a))
                        types.append(1 if m < 4 else 2)
        box = (L,L,L); tilt = (0,0,0)
    elif kind == "hcp":
        c = a*covera; cy = a*np.sqrt(3)/2
        pts, types = [], []
        for i in range(n):
            for j in range(n):
                for k in range(nz):
                    for fx, fy, fz in ((0,0,0),(1/3,2/3,.5)):
                        pts.append(((i+fx)*a+(j+fy)*(-a/2), (j+fy)*cy, (k+fz)*c))
                        types.append(1)
        box = (n*a, n*cy, nz*c); tilt = (-a/2, 0, 0)
    elif kind == "hex":            # WC-type: W at (0,0,0),(2/3,1/3,1/2); C at (1/3,2/3,1/2),(0,0,1/2)
        c = a*covera; cy = a*np.sqrt(3)/2
        pts, types = [], []
        for i in range(n):
            for j in range(n):
                for k in range(nz):
                    for fx, fy, fz, t in ((0,0,0,1),(2/3,1/3,.5,1),
                                          (1/3,2/3,.5,2),(0,0,.5,2)):
                        pts.append(((i+fx)*a+(j+fy)*(-a/2), (j+fy)*cy, (k+fz)*c))
                        types.append(t)
        box = (n*a, n*cy, nz*c); tilt = (-a/2, 0, 0)
    else:
        raise ValueError(kind)
    return pts, types, box, tilt

def energy(kind, els, a, covera=None):
    pts, types, box, tilt = cell(kind, els, a, covera)
    m = [MASS[e] for e in els]
    L = ["cap", "", f"{len(pts)} atoms", f"{len(els)} atom types", "",
         f"0 {box[0]:.6f} xlo xhi", f"0 {box[1]:.6f} ylo yhi", f"0 {box[2]:.6f} zlo zhi"]
    if any(abs(t) > 1e-9 for t in tilt):
        L.append(f"{tilt[0]:.6f} {tilt[1]:.6f} {tilt[2]:.6f} xy xz yz")
    L += ["", "Masses", ""]
    for i, mm in enumerate(m, 1):
        L.append(f"{i} {mm}")
    L += ["", "Atoms # atomic", ""]
    for i, (p, t) in enumerate(zip(pts, types), 1):
        L.append(f"{i} {t} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}")
    (W/"c.data").write_text("\n".join(L)+"\n", encoding="utf-8")
    inp = f"""units metal
atom_style atomic
read_data c.data
pair_style nep
pair_coeff * * {MODEL} {' '.join(els)}
neighbor 2.0 bin
neigh_modify one 4000
thermo_style custom step pe
minimize 1e-12 1e-12 5000 50000
print "EPER $(pe)"
"""
    (W/"in.e").write_text(inp, encoding="utf-8")
    subprocess.run([EXE,"-in","in.e","-log","log.e","-screen","none"], cwd=W,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    txt = (W/"log.e").read_text(errors="replace")
    mm = re.findall(r"EPER\s+(-?\d+\.\d+)", txt)
    n = len(pts)
    return (float(mm[-1])/n if mm else None), n

# (label, kind, elements, experimental a0, covera)
CASES = [
    ("C  diamond",   "diamond", ("C",),           3.567, None),
    ("Cu fcc",       "fcc",     ("Cu",),          3.615, None),
    ("Ti hcp",       "hcp",     ("Ti",),          2.951, 1.587),
    ("Cr bcc",       "bcc",     ("Cr",),          2.884, None),
    ("Mo bcc",       "bcc",     ("Mo",),          3.147, None),
    ("W  bcc",       "bcc",     ("W",),           3.165, None),
    ("TiC rocksalt", "rocksalt",("Ti","C"),       4.328, None),
    ("CrC rocksalt", "rocksalt",("Cr","C"),       4.13,  None),
    ("MoC rocksalt", "rocksalt",("Mo","C"),       4.28,  None),
    ("WC hex",       "hex",     ("W","C"),        2.906, 0.976),
]

print(f"{'phase':14s} {'a0':>7s} {'exp':>7s} {'dev%':>7s} {'E0/atom':>10s} {'conv':>6s} {'insd':>6s}  verdict")
results = {}
for label, kind, els, aexp, cr in CASES:
    xs, ys = [], []
    for f in np.arange(0.92, 1.081, 0.01):
        a = aexp*f
        e, n = energy(kind, els, a, cr)
        if e is not None:
            xs.append(a); ys.append(e)
    if len(xs) < 8:
        print(f"{label:14s}  --- calibration failed ---"); continue
    xs = np.array(xs); ys = np.array(ys)
    co = np.polyfit(xs, ys, 2)
    convex = co[0] > 0
    vertex = -co[1]/(2*co[0])
    inside = xs.min() < vertex < xs.max()
    e0 = float(np.polyval(co, vertex))
    dev = 100*(vertex-aexp)/aexp
    if not convex:
        verdict = "NO MINIMUM"
    elif not inside:
        verdict = "EDGE (min outside scan)"
    elif abs(dev) < 2.0:
        verdict = "USABLE"
    elif abs(dev) < 5.0:
        verdict = "marginal"
    else:
        verdict = "POOR"
    results[label] = dict(a0=round(vertex,4), e0=round(e0,4), dev=round(dev,2),
                          convex=bool(convex), inside=bool(inside), verdict=verdict)
    print(f"{label:14s} {vertex:7.3f} {aexp:7.3f} {dev:+7.2f} {e0:10.3f} "
          f"{str(convex):>6s} {str(inside):>6s}  {verdict}")

import json
(ROOT/"reports/capability_matrix.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print("\n[OK] reports/capability_matrix.json")
