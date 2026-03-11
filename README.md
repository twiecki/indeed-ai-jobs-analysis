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
- Sensitivity analysis discussion
