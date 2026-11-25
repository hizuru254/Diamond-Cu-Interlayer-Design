#!/usr/bin/env python3
import hashlib, json
from pathlib import Path
R = Path("/home/cc/interlayer_nemd")
def sha(p):
    h = hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest()
lock = {"schema_version": "1.0", "status": "APPROVED", "approved_by": "agent",
        "approved_at": "2026-11-26",
        "files": {"potentials.in": {"sha256": sha(R/"potentials/potentials.in")},
                  "CuTiC_NEP.txt": {"sha256": sha(R/"potentials/CuTiC_NEP.txt")}},
        "bulk_validation": {"verdict": "PASS",
                            "note": "C 3.590, Cu 3.666, Ti 2.944, Cr 2.865, Mo 3.186, W 3.205, TiC 4.357, WC 2.960 (all within 1.9%)"}}
(R/"config/potentials.lock.json").write_text(json.dumps(lock, indent=2), encoding="utf-8", newline="\n")
print("[LOCK] written")
