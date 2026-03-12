"""
Causal Impact Analysis: Are SWE job postings rising beyond what overall trends explain?

Uses CausalPy's Synthetic Control method with overall Indeed postings as control
to determine if software engineering postings show a genuine divergence.

Intervention date: ~Dec 2025 (when SWE postings start diverging upward)
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import causalpy as cp
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")

# Load data
swe = pd.read_csv("data/swe_postings.csv", parse_dates=["observation_date"])
overall = pd.read_csv("data/overall_postings.csv", parse_dates=["observation_date"])

# Merge
df = swe.merge(overall, on="observation_date", how="inner")
df.columns = ["date", "swe", "overall"]
df = df.set_index("date")
df = df.dropna()

# Focus on 2024 onwards (like the chart)
df = df.loc["2024-01-01":]

# Resample to weekly to smooth noise and speed up inference
df_weekly = df.resample("W").mean().dropna()

# Define intervention date: Dec 1, 2025 (where SWE starts diverging up)
intervention_date = pd.Timestamp("2025-12-01")

# Add time index for the model
df_weekly = df_weekly.reset_index()
df_weekly["t"] = np.arange(len(df_weekly))
df_weekly = df_weekly.set_index("date")

print(f"Data range: {df_weekly.index[0]} to {df_weekly.index[-1]}")
print(f"Pre-intervention: {len(df_weekly[df_weekly.index < intervention_date])} weeks")
print(f"Post-intervention: {len(df_weekly[df_weekly.index >= intervention_date])} weeks")
print(f"\nSWE last value: {df_weekly['swe'].iloc[-1]:.1f}")
print(f"Overall last value: {df_weekly['overall'].iloc[-1]:.1f}")

# Quick correlation check
pre = df_weekly[df_weekly.index < intervention_date]
post = df_weekly[df_weekly.index >= intervention_date]
print(f"\nPre-intervention correlation (SWE vs Overall): {pre['swe'].corr(pre['overall']):.3f}")
print(f"Post-intervention correlation (SWE vs Overall): {post['swe'].corr(post['overall']):.3f}")

# --- Plot 1: Raw data with intervention line ---
# Normalize both series to 100 at start for direct visual comparison
swe_norm = df_weekly["swe"] / df_weekly["swe"].iloc[0] * 100
overall_norm = df_weekly["overall"] / df_weekly["overall"].iloc[0] * 100

fig, ax = plt.subplots(figsize=(11, 5.5))
fig.patch.set_facecolor("white")
ax.set_facecolor("#FAFAFA")

# Shaded region for post-intervention
ax.axvspan(intervention_date, df_weekly.index[-1], alpha=0.08, color="#E74C3C", zorder=0)

# Lines
ax.plot(df_weekly.index, swe_norm, color="#E74C3C", linewidth=2.5, label="Software Engineering", zorder=3)
ax.plot(df_weekly.index, overall_norm, color="#7F8C8D", linewidth=2.5, label="All Job Postings", zorder=3)

# Intervention line
ax.axvline(intervention_date, color="#E74C3C", linestyle=":", linewidth=1.2, alpha=0.6, zorder=2)
ax.text(intervention_date + pd.Timedelta(days=5), ax.get_ylim()[1] * 0.99,
        "AI coding tools\ncross threshold", fontsize=9, color="#E74C3C", alpha=0.8,
        va="top", ha="left", fontstyle="italic")

# Axes
ax.set_ylabel("Job Postings Index (Jan 2024 = 100)", fontsize=11, color="#2C3E50")
ax.set_xlabel("")
ax.tick_params(colors="#7F8C8D", labelsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#D5D8DC")
ax.spines["bottom"].set_color("#D5D8DC")
ax.grid(axis="y", alpha=0.3, color="#BDC3C7")

# Legend
ax.legend(loc="lower left", frameon=True, framealpha=0.9, edgecolor="#D5D8DC",
          fontsize=10, fancybox=False)

# Title
ax.set_title("Software Engineering Job Postings Are Diverging Upward",
             fontsize=13, fontweight="bold", color="#2C3E50", pad=12)

plt.tight_layout()
plt.savefig("data/raw_comparison.png", dpi=200, facecolor="white", bbox_inches="tight")
plt.close()
print("\nSaved: data/raw_comparison.png")

# --- CausalPy Interrupted Time Series ---
print("\n--- Running CausalPy Interrupted Time Series ---")

# Use ITS with overall postings as predictor
# This fits a model pre-intervention and forecasts what SWE *would have been*
# if the relationship to overall postings stayed the same
df_cp = df_weekly[["swe", "overall"]].copy()

result = cp.InterruptedTimeSeries(
    df_cp,
    treatment_time=intervention_date,
    formula="swe ~ 1 + overall",
    model=cp.pymc_models.LinearRegression(
        sample_kwargs={"cores": 1, "chains": 2, "draws": 2000, "tune": 1000}
    ),
)

# Plot
fig, axes = result.plot()
fig.suptitle("Causal Impact: SWE Job Postings\n(controlling for overall job market)", y=1.02, fontsize=14)
fig.tight_layout()
fig.savefig("data/causal_impact.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: data/causal_impact.png")

# Summary
print("\n--- Summary ---")
print(result.summary())
