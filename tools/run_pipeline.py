#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path

def run(cmd: list[str]) -> None:
    print("\n$", " ".join(cmd))
    subprocess.check_call(cmd)

def main():
    repo = Path(__file__).resolve().parents[1]

    run([sys.executable, str(repo / "tools" / "gbif_incremental_update.py")])

    # Run your exporter (rename to whatever file you use)
    run([
        sys.executable,
        str(repo / "tools" / "export_occurrences_compact.py"),
        "--db", str(repo / "data" / "dwca.sqlite"),
        "--out", str(repo / "data" / "occurrences_compact.json"),
        "--country", "DE",
        "--year-from", "2021",
        "--year-to", "2025",
        "--top-n", "50",
        "--points-sample", "300",
        "--geohash-precision", "6"
    ])

if __name__ == "__main__":
    main()
