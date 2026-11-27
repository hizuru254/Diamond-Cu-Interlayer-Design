#!/usr/bin/env python3
"""Solution energy of C in Cu - the decisive check on the potential.

E_sol = E(Cu + C at a site) - E(pure Cu) - mu_C        mu_C = E/atom in diamond

Experimentally carbon is essentially insoluble in copper, so a physically sound
potential must give E_sol > 0 (by ~0.5-1 eV).  A strongly NEGATIVE value would
mean this NEP over-binds C-Cu, which would explain why every diamond/Cu interface
we simulate dissolves.
"""
from __future__ import annotations
import re, subprocess
from pathlib import Path
import numpy as np

EXE = "/home/cc/lammps/src/lmp_mpi"
MODEL = "/home/cc/interlayer_nemd/potentials/CuTiC_NEP.txt"
W = Path("/tmp/csol"); W.mkdir(parents=True, exist_ok=True)
A0 = 3.666          # NEP equilibrium fcc Cu

def fcc(n=3):
    b = [(0,0,0),(0,.5,.5),(.5,0,.5),(.5,.5,0)]
    pts = [((i+bx)*A0,(j+by)*A0,(k+bz)*A0) for i in range(n) for j in range(n)
           for k in range(n) for bx,by,bz in b]
    return np.array(pts), n*A0

def run(name, pts, types):
    L = A0*3
    lines = [name, "", f"{len(pts)} atoms", "2 atom types", "",
             f"0 {L:.6f} xlo xhi", f"0 {L:.6f} ylo yhi", f"0 {L:.6f} zlo zhi", "",
             "Masses", "", "1 12.011", "2 63.546", "", "Atoms # atomic", ""]
    for i,(p,t) in enumerate(zip(pts,types),1):
        lines.append(f"{i} {t} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}")
    (W/"c.data").write_text("\n".join(lines)+"\n", encoding="utf-8")
    inp = f"""units metal
atom_style atomic
read_data c.data
pair_style nep
pair_coeff * * {MODEL} C Cu
neighbor 2.0 bin
neigh_modify one 4000
thermo_style custom step pe
minimize 1e-10 1e-10 4000 40000
print "EPER $(pe)"
"""
    (W/"in.e").write_text(inp, encoding="utf-8")
    subprocess.run([EXE,"-in","in.e","-log","log.e","-screen","none"], cwd=W,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t = (W/"log.e").read_text(errors="replace")
    m = re.findall(r"EPER\s+(-?\d+\.\d+)", t)
    return float(m[-1])

pts, L = fcc(3)
n_cu = len(pts)                      # 108
E_pure = run("pure Cu", pts, [2]*n_cu)

# octahedral interstitial at the cube centre
oct_site = np.array([[A0*1.5, A0*1.5, A0*1.5]])
pts_i = np.vstack([pts, oct_site])
E_int = run("C interstitial", pts_i, [2]*n_cu + [1])

# substitutional: replace the Cu closest to that point
idx = int(np.argmin(np.linalg.norm(pts - oct_site[0], axis=1)))
pts_s = pts.copy(); types_s = [2]*n_cu; types_s[idx] = 1
E_sub = run("C substitutional", pts_s, types_s)

# diamond reference (NEP value from the earlier calibration: 8-atom cell)
def diamond_ref():
    b = [(0,0,0),(0,.5,.5),(.5,0,.5),(.5,.5,0),
         (.25,.25,.25),(.25,.75,.75),(.75,.25,.75),(.75,.75,.25)]
    a = 3.590
    pts = np.array([[bx*a,by*a,bz*a] for bx,by,bz in b])
    lines = ["diamond","", "8 atoms","1 atom types","",f"0 {a} xlo xhi",f"0 {a} ylo yhi",f"0 {a} zlo zhi","",
             "Masses","","1 12.011","","Atoms # atomic",""]
    for i,p in enumerate(pts,1):
        lines.append(f"{i} 1 {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}")
    (W/"d.data").write_text("\n".join(lines)+"\n", encoding="utf-8")
    inp = f"""units metal
atom_style atomic
read_data d.data
pair_style nep
pair_coeff * * {MODEL} C
neighbor 2.0 bin
thermo_style custom step pe
minimize 1e-10 1e-10 4000 40000
print "EPER $(pe)"
"""
    (W/"in.d").write_text(inp, encoding="utf-8")
    subprocess.run([EXE,"-in","in.d","-log","log.d","-screen","none"], cwd=W,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t = (W/"log.d").read_text(errors="replace")
    m = re.findall(r"EPER\s+(-?\d+\.\d+)", t)
    return float(m[-1])/8.0

mu_C = diamond_ref()
print(f"E(pure Cu, {n_cu} atoms)      = {E_pure:12.3f} eV")
print(f"mu_C (diamond)               = {mu_C:12.3f} eV/atom")
print(f"E(Cu + C interstitial)       = {E_int:12.3f} eV")
print(f"E(Cu + C substitutional)     = {E_sub:12.3f} eV")
print(f"\nE_sol (interstitial) = {E_int - E_pure - mu_C:+.3f} eV")
print(f"E_sol (substitutional)= {E_sub - E_pure - mu_C:+.3f} eV")
print("\nExperiment: C is essentially insoluble in Cu -> E_sol should be clearly POSITIVE.")
