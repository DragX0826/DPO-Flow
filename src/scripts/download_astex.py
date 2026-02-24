#!/usr/bin/env python3
"""
Download the Astex Diverse Set 85 complexes from RCSB PDB.
Run BEFORE uploading to Kaggle:
    python scripts/download_astex.py --output data/astex_pdb/

Kaggle usage:
    1. Run this script locally
    2. Upload data/astex_pdb/ as a Kaggle dataset named "astex-diverse"
    3. Then: --pdb_dir /kaggle/input/astex-diverse
"""
import os
import sys
import time
import urllib.request
import argparse

ASTEX_DIVERSE_85 = [
    "1aq1","1b8o","1cvu","1d3p","1eve","1f0r","1fc0","1fpu","1glh",
    "1gpk","1hw8","1hwi","1ig3","1j3j","1jd0","1k3u","1ke5","1kzk",
    "1l2s","1lpg","1lpk","1m2z","1mq6","1n2v","1n46","1nav","1o3f",
    "1of1","1opk","1oq5","1owh","1p2y","1pxn","1q41","1q8t","1qkt",
    "1r1h","1r55","1s19","1s3v","1sg0","1sj0","1sqt","1t46","1tt1",
    "1u1c","1vso","1w1p","1xm6","2br1","2brl","2brn","2cet","2ch0",
    "2cji","2cnp","2cpp","2gss","2hs1","2i1m","2i78","2ica","2j4i",
    "2jcj","2jdm","2jdu","2jf4","2nlj","2nnq","2npo","2p15","2p4y",
    "2p54","2p55","2p7a","2pog","2psh","2pwy","2qmj","2wnc","2xnb",
    "2xys","2yge","2zjw",
]

RCSB_URL = "https://files.rcsb.org/download/{}.pdb"


def download_pdb(pdb_id, output_dir, retry=3):
    out_path = os.path.join(output_dir, f"{pdb_id}.pdb")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
        print(f"  SKIP {pdb_id} (already downloaded)")
        return True
    url = RCSB_URL.format(pdb_id.upper())
    for attempt in range(retry):
        try:
            urllib.request.urlretrieve(url, out_path)
            size_kb = os.path.getsize(out_path) / 1024
            print(f"  OK   {pdb_id}  ({size_kb:.0f} KB)")
            return True
        except Exception as e:
            wait = 2 ** attempt
            print(f"  FAIL {pdb_id} attempt {attempt+1}: {e}. Retrying in {wait}s")
            time.sleep(wait)
    print(f"  ERROR {pdb_id}: all retries failed")
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/astex_pdb", help="Output directory")
    parser.add_argument("--targets", nargs="+", default=ASTEX_DIVERSE_85,
                        help="Subset of PDB IDs (default: all 85)")
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    print(f"Downloading {len(args.targets)} Astex Diverse Set PDB files → {args.output}/")
    ok = sum(download_pdb(pid, args.output) for pid in args.targets)
    print(f"\nDone: {ok}/{len(args.targets)} downloaded to {args.output}/")
    if ok < len(args.targets):
        sys.exit(1)


if __name__ == "__main__":
    main()
