#!/usr/bin/env python3
"""Build diamond(111)/interlayer/Cu(111) with per-material commensuration.

In-plane period L is shared, but each material uses its own number of hexagonal
cells so that no material is over-strained:

    diamond  6 cells: a = L/6   (bulk NN 2.522 A)
    Cu       6 cells: a = L/6   (bulk NN 2.581 A)
    TiC      5 cells: a = L/5   (bulk NN 3.081 A)

L = 15.31 A gives +1.2 % / -1.2 % / -0.6 % strain.  Stacking along z:
  diamond (111): bilayers, the second atom of a pair sits 1.545 A straight above
  TiC (111): alternating Ti and C planes, d = a/(2 sqrt3)
  Cu (111): ABC, d = a0/sqrt3
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

A_DIA, A_CU0, A_TIC = 3.567, 3.650, 4.357
NN = {"C": A_DIA/np.sqrt(2.0), "Cu": A_CU0/np.sqrt(2.0), "TiC": A_TIC/np.sqrt(2.0)}
L = 15.31
LY = L*np.sqrt(3.0)
TYPE = {"C": 1, "Ti": 2, "Cu": 3, "W": 4, "Cr": 5, "Mo": 6}
MASS = {"C": 12.011, "Ti": 47.867, "Cu": 63.546, "W": 183.84, "Cr": 52.00, "Mo": 95.95}

def tri(n, a, shift=(0.0, 0.0)):
    pts = []
    for i in range(n):
        for j in range(n):
            for bx, by in ((0.0, 0.0), (0.5, 0.5)):
                pts.append((((i+bx)*a + shift[0]) % (n*a),
                            ((j+by)*a*np.sqrt(3) + shift[1]) % (n*a*np.sqrt(3))))
    return np.array(pts)

def stack_layers(n, a, nz, z0, dz, shifts, elements):
    """shifts: list of in-plane offsets per sub-layer; elements: name per sub-layer."""
    out, tps = [], []
    for k in range(nz):
        for s, (sh, el) in enumerate(zip(shifts, elements)):
            p = tri(n, a, sh)
            out.append(np.hstack([p, np.full((len(p), 1), z0 + k*dz + s*_sub_spacing)]))
            tps.append(np.full(len(p), TYPE[el]))
    return np.vstack(out), np.concatenate(tps)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", required=True)
    ap.add_argument("--out-structure", required=True)
    ap.add_argument("--out-geom", required=True)
    ap.add_argument("--shift1", default="0,0", help="lateral shift of interlayer vs diamond (A)")
    ap.add_argument("--shift2", default="0,0", help="lateral shift of Cu vs interlayer (A)")
    a = ap.parse_args()
    spec = json.loads(Path(a.design).read_text(encoding="utf-8"))
    mat = spec.get("interlayer", "TiC")
    def get_A(key, default_nm):
        """Thickness in Angstrom. Accepts key_A (preferred) or key_nm (x10)."""
        if key + "_A" in spec:
            return float(spec[key + "_A"])
        if key + "_nm" in spec:
            return float(spec[key + "_nm"]) * 10.0
        return default_nm * 10.0

    t_nm = get_A("thickness", 2.0) / 10.0      # interlayer thickness in nm (legacy var name)
    term = spec.get("termination", "Ti")
    term_cu = spec.get("termination_cu", None)
    cu_nm = get_A("cu_thickness", 60.0) / 10.0
    dia_nm = get_A("diamond_thickness", 60.0) / 10.0
    vac = get_A("vacuum", 1.5) / 10.0
    print(f"[spec] interlayer {t_nm*10:.1f} A, diamond {dia_nm*10:.1f} A, "
          f"Cu {cu_nm*10:.1f} A, vacuum {vac*10:.1f} A")
    if mat not in ("TiC", "none"):
        raise SystemExit(f"{mat}: only TiC / WC / none enabled so far")

    n_d = n_c = 6; n_i = 5
    a_d, a_c, a_i = L/n_d, L/n_c, L/n_i
    for name, aa, ref in (("diamond", a_d, NN["C"]), ("Cu", a_c, NN["Cu"]), ("TiC", a_i, NN["TiC"])):
        print(f"[strain] {name:8s} a={aa:.3f} vs bulk {ref:.3f}  {100*(aa-ref)/ref:+.2f}%")

    s1 = [float(x) for x in a.shift1.split(",")]
    s2 = [float(x) for x in a.shift2.split(",")]
    parts, types = [], []

    # --- diamond (bottom) ----------------------------------------------
    dz_d = A_DIA/np.sqrt(3.0)          # 2.059 A between (111) planes
    sub = 1.545                         # C-C bond along [111]
    nz_d = int(round(dia_nm*10.0/dz_d))
    pts_d, tps_d = [], []
    for k in range(nz_d):
        s = k % 3
        sh = (s*a_d/2.0, s*a_d/(2.0*np.sqrt(3)))
        p = tri(n_d, a_d, sh)
        pts_d.append(np.hstack([p, np.full((len(p), 1), k*dz_d)]))
        pts_d.append(np.hstack([p, np.full((len(p), 1), k*dz_d + sub)]))
        tps_d.append(np.full(2*len(p), TYPE["C"]))
    dia = np.vstack(pts_d); dia[:, 2] -= dia[:, 2].max()
    parts.append(dia); types.append(np.concatenate(tps_d))

    # --- interlayer ------------------------------------------------------
    gap = 2.10
    lay = None
    if mat == "none":
        print("[spec] no interlayer: direct diamond/Cu contact")
        z_top = 0.0
    elif mat == "TiC":
        d_i = A_TIC/(2.0*np.sqrt(3.0))
        nz_i = max(2, int(round(t_nm*10.0/d_i)))
        if term_cu is not None:
            want_odd = (term_cu == term)
            if want_odd and nz_i % 2 == 0:
                nz_i += 1
            elif (not want_odd) and nz_i % 2 == 1:
                nz_i += 1
            print(f"[spec] interlayer planes {nz_i} -> bottom {term}-term, "
                  f"top {(term if nz_i % 2 else ('C' if term=='Ti' else 'Ti'))}-term")
        pts_i, tps_i = [], []
        for k in range(nz_i):
            s = k % 3
            sh = (s*a_i/2.0, s*a_i/(2.0*np.sqrt(3)))
            el = term if k % 2 == 0 else ("C" if term == "Ti" else "Ti")
            pp = tri(n_i, a_i, sh)
            pts_i.append(np.hstack([pp, np.full((len(pp), 1), gap + k*d_i)]))
            tps_i.append(np.full(len(pp), TYPE[el]))
        lay = np.vstack(pts_i)
        lay[:, 0] = (lay[:, 0] + s1[0]) % L
        lay[:, 1] = (lay[:, 1] + s1[1]) % LY
        parts.append(lay); types.append(np.concatenate(tps_i))
        z_top = lay[:, 2].max()
    else:
        raise SystemExit(f"{mat}: only TiC / WC / none enabled so far")

    # --- Cu (top) ---------------------------------------------------------
    dz_c = A_CU0/np.sqrt(3.0)
    nz_c = int(round(cu_nm*10.0/dz_c))
    pts_c, tps_c = [], []
    for k in range(nz_c):
        s = k % 3
        sh = (s*a_c/2.0, s*a_c/(2.0*np.sqrt(3)))
        p = tri(n_c, a_c, sh)
        pts_c.append(np.hstack([p, np.full((len(p), 1), z_top + 2.10 + k*dz_c)]))
        tps_c.append(np.full(len(p), TYPE["Cu"]))
    cu = np.vstack(pts_c)
    cu[:, 0] = (cu[:, 0] + s1[0] + s2[0]) % L
    cu[:, 1] = (cu[:, 1] + s1[1] + s2[1]) % LY
    parts.append(cu); types.append(np.concatenate(tps_c))

    atoms = np.vstack(parts); tps = np.concatenate(types)
    zmin, zmax = atoms[:,2].min()-vac*10, atoms[:,2].max()+vac*10

    def minimg(ia, ib, same=False):
        pa, pb = atoms[ia], atoms[ib]; best = 1e9
        for i in range(0, len(pa), 512):
            dx = np.abs(pa[i:i+512,None,0]-pb[None,:,0]); dx = np.minimum(dx, L-dx)
            dy = np.abs(pa[i:i+512,None,1]-pb[None,:,1]); dy = np.minimum(dy, LY-dy)
            dz = pa[i:i+512,None,2]-pb[None,:,2]
            d = np.sqrt(dx*dx+dy*dy+dz*dz)
            if same:
                for k in range(d.shape[0]): d[k, i+k] = np.inf
            best = min(best, d.min())
        return best

    ok = True
    for name, idx in (("C", tps==1), ("Ti", tps==2), ("Cu", tps==3)):
        if idx.sum() > 1:
            dd = minimg(np.where(idx)[0], np.where(idx)[0], same=True)
            flag = "OK" if dd > 1.35 else "TOO SHORT"
            if flag != "OK": ok = False
            print(f"[check] {name}-{name} min {dd:.3f} A  {flag}")
    ic = np.where(tps==1)[0]; im = np.where(tps!=1)[0]
    dd = minimg(im, ic)
    print(f"[check] metal-C min {dd:.3f} A  {'OK' if dd > 1.6 else 'TOO SHORT'}")
    if not ok:
        print("[ABORT] geometry has overlapping atoms"); return 1

    lines = ["diamond/TiC/Cu", "", f"{len(atoms)} atoms", "3 atom types", "",
             f"0 {L:.6f} xlo xhi", f"0 {LY:.6f} ylo yhi",
             f"{zmin:.6f} {zmax:.6f} zlo zhi", "", "Masses", ""]
    for t, e in ((1,"C"),(2,"Ti"),(3,"Cu")):
        lines.append(f"{t} {MASS[e]}")
    lines += ["", "Atoms # atomic", ""]
    for i, (p, t) in enumerate(zip(atoms, tps), 1):
        lines.append(f"{i} {t} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}")
    Path(a.out_structure).write_text("\n".join(lines)+"\n", encoding="utf-8")
    # ---- geom for preflight ------------------------------------------
    z_dia_top = float(dia[:,2].max())
    z_cu_bot = float(cu[:,2].min())
    zi = 0.5*(z_dia_top + z_cu_bot)
    if lay is None:
        layers = [{"id": "DIA", "material": "C", "z_min": float(dia[:,2].min()), "z_max": zi},
                  {"id": "CU", "material": "Cu", "z_min": zi, "z_max": float(cu[:,2].max())}]
        ifaces = [{"id": "DIA_CU", "z": zi,
                   "left": {"z_min": zi-15.0, "z_max": zi-2.0, "material": "C"},
                   "right": {"z_min": zi+2.0, "z_max": zi+15.0, "material": "Cu"}}]
    else:
        z_tic_top = float(lay[:,2].max())
        i2 = 0.5*(z_tic_top + z_cu_bot)
        layers = [{"id": "DIA", "material": "C", "z_min": float(dia[:,2].min()), "z_max": zi},
                  {"id": "INT", "material": mat, "z_min": zi, "z_max": i2},
                  {"id": "CU", "material": "Cu", "z_min": i2, "z_max": float(cu[:,2].max())}]
        ifaces = [{"id": "DIA_INT", "z": zi,
                   "left": {"z_min": zi-10.0, "z_max": zi-2.0, "material": "C"},
                   "right": {"z_min": zi+2.0, "z_max": zi+10.0, "material": mat}},
                  {"id": "INT_CU", "z": i2,
                   "left": {"z_min": i2-10.0, "z_max": i2-2.0, "material": mat},
                   "right": {"z_min": i2+2.0, "z_max": i2+10.0, "material": "Cu"}}]
    geom = {
      "schema_version": "1.0", "status": "APPROVED",
      "geometry_id": f"STACK_{mat}",
      "note": f"diamond(111)/{mat}/Cu(111), common L={L:.3f} A",
      "box": {"lx": L, "ly": LY, "z_min": zmin, "z_max": zmax, "boundary": "p p p"},
      "layers": layers, "interfaces": ifaces,
      "groups": {
        "hot":  {"z_min": float(dia[:,2].min())+6.0, "z_max": float(dia[:,2].min())+18.0},
        "cold": {"z_min": float(cu[:,2].max())-22.0, "z_max": float(cu[:,2].max())},
      },
      "chunk_bin_A": 1.0,
      "expected_atoms": {"total": int(len(atoms))},
    }
    Path(a.out_geom).write_text(json.dumps(geom, indent=2, ensure_ascii=False)+"\n",
                               encoding="utf-8", newline="\n")
    print(f"[geom] {len(layers)} layers, {len(ifaces)} interface(s) at z={zi:.2f}")

    print(f"[built] {len(atoms)} atoms  box {L:.2f} x {LY:.2f} x {zmax-zmin:.1f}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
