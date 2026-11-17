"""Compares two flat2rig-cli output folders file by file (evaluation helper).

Every file must be byte-identical except stats.json, which is compared as JSON with the
run-dependent `elapsedMs` removed (top level and inside `fidelity.holdout`). Used by T14 to show that
the default CLI run is unchanged by the evaluation-only code paths.

  tools/evaluation/.venv/Scripts/python.exe -I tools/evaluation/compare_cli_outputs.py DIR_A DIR_B
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def strip(v):
    if isinstance(v, dict):
        return {k: strip(x) for k, x in v.items() if k != "elapsedMs"}
    if isinstance(v, list):
        return [strip(x) for x in v]
    return v


def main() -> int:
    a, b = Path(sys.argv[1]), Path(sys.argv[2])
    fa = sorted(p.relative_to(a).as_posix() for p in a.rglob("*") if p.is_file())
    fb = sorted(p.relative_to(b).as_posix() for p in b.rglob("*") if p.is_file())
    if fa != fb:
        print("file sets differ:", sorted(set(fa) ^ set(fb)))
        return 1
    bad = []
    for rel in fa:
        da, db = (a / rel).read_bytes(), (b / rel).read_bytes()
        if rel == "stats.json":
            same = strip(json.loads(da)) == strip(json.loads(db))
        else:
            same = da == db
        if not same:
            bad.append(rel)
    digest = hashlib.sha256()
    for rel in fa:
        if rel != "stats.json":
            digest.update(rel.encode() + b"\0" + hashlib.sha256((a / rel).read_bytes()).digest())
    print(f"{len(fa)} files compared, {len(bad)} differ; non-stats tree sha256 {digest.hexdigest()}")
    for rel in bad:
        print("  differs:", rel)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
