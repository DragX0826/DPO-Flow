#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd


def resolve_csvs(pattern: str) -> list[str]:
    matches = sorted(glob.glob(pattern, recursive=True))
    out: list[str] = []
    for path in matches:
        if os.path.isdir(path):
            out.extend(sorted(glob.glob(os.path.join(path, "**", "benchmark_results.csv"), recursive=True)))
        elif path.lower().endswith(".csv"):
            out.append(path)
    if not out and os.path.isdir(pattern):
        out = sorted(glob.glob(os.path.join(pattern, "**", "benchmark_results.csv"), recursive=True))
    if not out and os.path.isfile(pattern):
        out = [pattern]
    return sorted(set(out))


def classify_target(search_gap: float, selection_gap: float, oracle_rmsd: float) -> str:
    if np.isnan(oracle_rmsd):
        return "unknown"
    if oracle_rmsd > 4.0:
        return "search_limited"
    if selection_gap >= 0.25:
        return "ranking_limited"
    if search_gap <= 0.25 and selection_gap <= 0.25:
        return "well_aligned"
    return "mixed"


def main():
    parser = argparse.ArgumentParser(description="Audit search-vs-selection gaps from benchmark_results.csv files.")
    parser.add_argument("--run", action="append", required=True, help='Format: name="glob_or_csv"')
    parser.add_argument("--targets", type=str, default="", help="Optional comma-separated target subset")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for audit outputs")
    args = parser.parse_args()

    targets = {t.strip().lower() for t in args.targets.split(",") if t.strip()}
    frames = []
    for item in args.run:
        if "=" not in item:
            raise SystemExit(f"Invalid --run entry: {item}")
        name, pattern = item.split("=", 1)
        name = name.strip()
        pattern = pattern.strip().strip('"').strip("'")
        csvs = resolve_csvs(pattern)
        for csv_path in csvs:
            df = pd.read_csv(csv_path)
            if df.empty or "pdb_id" not in df.columns:
                continue
            df = df.copy()
            df["method"] = name
            frames.append(df)

    if not frames:
        raise SystemExit("No benchmark CSVs resolved")

    df = pd.concat(frames, ignore_index=True)
    df["pdb_id"] = df["pdb_id"].astype(str).str.lower()
    if targets:
        df = df[df["pdb_id"].isin(targets)].copy()

    for col in ["best_rmsd", "oracle_best_rmsd", "mean_rmsd", "ranked_rmsd", "mmff_fallback_rate"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    grouped = (
        df.groupby(["method", "pdb_id"], as_index=False)
        .agg(
            selected_rmsd=("best_rmsd", "mean"),
            oracle_rmsd=("oracle_best_rmsd", "mean"),
            mean_rmsd=("mean_rmsd", "mean"),
            ranked_rmsd=("ranked_rmsd", "mean"),
            fallback_rate=("mmff_fallback_rate", "mean"),
            n_seeds=("seed", "nunique") if "seed" in df.columns else ("pdb_id", "size"),
        )
    )
    grouped["search_gap"] = grouped["oracle_rmsd"] - grouped["selected_rmsd"]
    grouped["selection_gap"] = grouped["selected_rmsd"] - grouped["oracle_rmsd"]
    grouped["ranking_gap"] = grouped["ranked_rmsd"] - grouped["oracle_rmsd"]
    grouped["target_class"] = [
        classify_target(search_gap, selection_gap, oracle)
        for search_gap, selection_gap, oracle in zip(grouped["search_gap"], grouped["selection_gap"], grouped["oracle_rmsd"])
    ]

    summary = (
        grouped.groupby(["method", "target_class"], as_index=False)
        .agg(
            n_targets=("pdb_id", "count"),
            mean_selected_rmsd=("selected_rmsd", "mean"),
            mean_oracle_rmsd=("oracle_rmsd", "mean"),
            mean_selection_gap=("selection_gap", "mean"),
            mean_ranking_gap=("ranking_gap", "mean"),
            mean_fallback_rate=("fallback_rate", "mean"),
        )
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(out_dir / "target_gap_audit.csv", index=False)
    summary.to_csv(out_dir / "target_gap_summary.csv", index=False)

    md_lines = ["# Search-vs-Selection Gap Audit", ""]
    for method, g in grouped.groupby("method", sort=False):
        md_lines.append(f"## {method}")
        for _, row in g.sort_values(["target_class", "selection_gap"], ascending=[True, False]).iterrows():
            md_lines.append(
                f"- `{row['pdb_id']}`: selected={row['selected_rmsd']:.3f} A, "
                f"oracle={row['oracle_rmsd']:.3f} A, selection_gap={row['selection_gap']:.3f} A, "
                f"ranked_rmsd={row['ranked_rmsd']:.3f} A, class={row['target_class']}"
            )
        md_lines.append("")
    (out_dir / "target_gap_audit.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote audit to {out_dir}")


if __name__ == "__main__":
    main()
