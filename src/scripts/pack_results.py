#!/usr/bin/env python3
"""
pack_results.py — Auto-package experiment results into a timestamped zip archive.

Usage:
    python scripts/pack_results.py                          # pack everything
    python scripts/pack_results.py --label my_run_v1       # custom label
    python scripts/pack_results.py --output /kaggle/working # Kaggle output dir

Output: results_YYYYMMDD_HHMMSS_<label>.zip  containing:
  - results/benchmark_results.csv
  - results/*.pdb    (best poses)
  - plots/*.pdf/.png (all figures)
  - metadata.json    (git hash, timestamp, args)
"""
import os
import sys
import json
import zipfile
import hashlib
import datetime
import argparse
import subprocess
import glob


def git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(__file__),
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def collect_files(src_root):
    patterns = [
        "results/benchmark_results.csv",
        "results/*.pdb",
        "plots/*.pdf",
        "plots/*.png",
    ]
    files = []
    for pat in patterns:
        full_pat = os.path.join(src_root, pat)
        matched = glob.glob(full_pat)
        files.extend(matched)
    return sorted(set(files))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="", help="Custom label suffix")
    parser.add_argument("--output", default=".", help="Where to write the zip")
    parser.add_argument("--src", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        help="Project root (default: parent of scripts/)")
    args = parser.parse_args()

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    label = f"_{args.label}" if args.label else ""
    archive_name = f"saebflow_results_{ts}{label}.zip"
    archive_path = os.path.join(args.output, archive_name)

    files = collect_files(args.src)
    if not files:
        print("[pack_results] No output files found. Run the benchmark first.")
        sys.exit(1)

    # Build metadata
    meta = {
        "git_hash":    git_hash(),
        "timestamp":   ts,
        "label":       args.label,
        "num_files":   len(files),
        "files":       {},
    }

    os.makedirs(args.output, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fpath in files:
            arcname = os.path.relpath(fpath, args.src)
            zf.write(fpath, arcname)
            meta["files"][arcname] = sha256_file(fpath)
            print(f"  + {arcname}")

        # Write inline metadata
        zf.writestr("metadata.json", json.dumps(meta, indent=2))

    size_mb = os.path.getsize(archive_path) / 1e6
    print(f"\n[pack_results] Archive ready: {archive_path}  ({size_mb:.2f} MB)")
    print(f"  Files: {len(files)}  |  git: {meta['git_hash']}")
    return archive_path


if __name__ == "__main__":
    main()
