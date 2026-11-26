#!/usr/bin/env python3
"""Registry optimisation: scan lateral shifts of each interface, minimise, compare energy.

The two interfaces are treated sequentially (they are ~40 A apart, so the coupling
is weak): first shift1 with shift2 = 0, then shift2 with the best shift1.
Only the total static energy matters - the bulk contributions are identical.
"""
from __future__ import annotations
import argparse, json, re, subprocess, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EXE = "/home/cc/lammps/src/lmp_mpi"
W = Path("/tmp/reg"); W.mkdir(parents=True, exist_ok=True)


def static_energy(struct: Path, model: Path) -> float:
    inp = f"""units metal
atom_style atomic
read_data {struct}
pair_style nep
pair_coeff * * {model} C Ti Cu
neighbor 2.0 bin
neigh_modify one 4000
thermo_style custom step pe
minimize 1e-10 1e-10 3000 30000
print "EPER $(pe)"
"""
    (W/"in.e").write_text(inp, encoding="utf-8")
    subprocess.run([EXE, "-in", "in.e", "-log", "log.e", "-screen", "none"],
                   cwd=W, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t = (W/"log.e").read_text(errors="replace")
    m = re.findall(r"EPER\s+(-?\d+\.\d+)", t)
    return float(m[-1]) if m else float("nan")


def build(design: Path, s1, s2, out: Path) -> bool:
    r = subprocess.run([sys.executable, str(ROOT/"tools/build_stack.py"),
                        "--design", str(design), "--out-structure", str(out),
                        "--out-geom", "/tmp/reg/g.json",
                        "--shift1", f"{s1[0]},{s1[1]}", "--shift2", f"{s2[0]},{s2[1]}"],
                       capture_output=True, text=True)
    return r.returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", required=True)
    ap.add_argument("--model", default=str(ROOT/"potentials/CuTiC_NEP.txt"))
    ap.add_argument("--out", default="reports/registry.json")
    a = ap.parse_args()
    L, LY = 15.31, 15.31*np.sqrt(3)
    gx = [0.0, L/6, L/3, L/2]
    gy = [0.0, LY/6, LY/3]
    model = Path(a.model)

    def scan(fixed1, which):
        best = None
        for dx in gx:
            for dy in gy:
                s1 = (dx, dy) if which == 1 else fixed1
                s2 = (dx, dy) if which == 2 else (0.0, 0.0)
                if not build(Path(a.design), s1, s2, W/"s.data"):
                    continue
                e = static_energy(W/"s.data", model)
                if not np.isfinite(e):
                    continue
                if best is None or e < best[0]:
                    best = (e, dx, dy)
                print(f"  {which}: shift=({dx:5.2f},{dy:5.2f})  E={e:12.3f}", flush=True)
        return best

    print("=== interface 1: diamond / interlayer ===")
    b1 = scan((0.0, 0.0), 1)
    print(f"[best1] shift1=({b1[1]:.2f},{b1[2]:.2f})  E={b1[0]:.3f}")
    print("=== interface 2: interlayer / Cu ===")
    b2 = scan((b1[1], b1[2]), 2)
    print(f"[best2] shift2=({b2[1]:.2f},{b2[2]:.2f})  E={b2[0]:.3f}")
    out = {"shift1": [b1[1], b1[2]], "E1": b1[0], "shift2": [b2[1], b2[2]], "E2": b2[0]}
    (ROOT/a.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("[OK]", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
