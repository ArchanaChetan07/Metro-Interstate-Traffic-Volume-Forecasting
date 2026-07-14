#!/usr/bin/env python3
"""Thin entrypoint notebooks / explorers can call for package metrics.

Usage:
  PYTHONPATH=. python scripts/package_demo.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_benchmark.py"),
        "--out",
        str(ROOT / "artifacts" / "benchmark_report.json"),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    print("Running:", " ".join(cmd))
    rc = subprocess.call(cmd, cwd=str(ROOT), env=env)
    report = ROOT / "artifacts" / "benchmark_report.json"
    if report.exists():
        data = json.loads(report.read_text(encoding="utf-8"))
        print("\nKey R² values:")
        for name, m in data.get("models", {}).items():
            if "r2" in m:
                print(f"  {name}: {m['r2']:.4f}")
        adf = data.get("adf", {})
        print(f"  ADF: {adf.get('statistic'):.4f} (p={adf.get('pvalue')})")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
