# Indeed AI/SWE Job Postings Analysis

Interactive analysis of the divergence of Software Engineering job postings from overall job market trends, using [Indeed's Job Postings data from FRED](https://fred.stlouisfed.org/series/IHLIDXUSTPSOFTDEVE).

## Data Sources

- **IHLIDXUSTPSOFTDEVE** — Indeed Job Postings: Software Development (US)
- **IHLIDXUS** — Indeed Job Postings: Total (US)

Both series are indexed to 100 on Feb 1, 2020.

## Run

```bash
# Install dependencies
uv sync

# Launch the interactive notebook
uv run marimo edit indeed_ai_jobs_analysis.py
```

## What's in the notebook

- Raw data exploration with interactive table
- Normalized comparison chart (SWE vs All Postings)
- Gap analysis (SWE − Overall in percentage points)
- Pre/post correlation statistics
- CausalPy Interrupted Time Series analysis
- Sensitivity analysis discussion

## Standalone causal analysis script

`analyze_indeed_causal.py` runs the CausalPy analysis as a standalone script (no Marimo required) and generates publication-ready PNG plots:

```bash
uv run python analyze_indeed_causal.py
```

Outputs:
- `data/raw_comparison.png` — Normalized index comparison chart
- `data/causal_impact.png` — CausalPy ITS regression plot
