# Contributing

## Scope

This repository is a research codebase. Contributions should improve:

- benchmark reproducibility
- search or scoring logic
- report generation
- documentation clarity

## Setup

Use Python 3.10+.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Before opening a PR

- keep changes focused
- avoid committing generated results, caches, or local notebooks
- update docs when behavior changes
- regenerate `reports/` only when report content actually changes

## Validation

At minimum, run:

```bash
python -m compileall src scripts quantum
```

If your change affects benchmark aggregation or report generation, also run the relevant script locally and verify outputs under `reports/`.

## Style

- prefer small, reviewable changes
- keep public-facing wording neutral and precise
- avoid adding temporary benchmark artifacts to version control
