import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
        # Indeed AI/SWE Job Postings Analysis

        Analyzing the divergence of Software Engineering job postings from overall job market trends,
        using [Indeed's Job Postings data from FRED](https://fred.stlouisfed.org/series/IHLIDXUSTPSOFTDEVE).

        **Data sources:**
        - `IHLIDXUSTPSOFTDEVE` — Indeed Job Postings: Software Development (US)
        - `IHLIDXUS` — Indeed Job Postings: Total (US)
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    return go, make_subplots, mo, np, pd


@app.cell
def _(pd):
    # Load raw data from FRED/Indeed
    swe = pd.read_csv(
        "data/swe_postings.csv", parse_dates=["observation_date"]
    ).rename(columns={"IHLIDXUSTPSOFTDEVE": "swe", "observation_date": "date"})

    overall = pd.read_csv(
        "data/overall_postings.csv", parse_dates=["observation_date"]
    ).rename(columns={"IHLIDXUS": "overall", "observation_date": "date"})

    df = swe.merge(overall, on="date", how="inner").set_index("date").dropna()

    mo_stats = f"**{len(df):,} observations** from {df.index[0].date()} to {df.index[-1].date()}"
    return df, mo_stats, overall, swe


@app.cell
def _(mo, mo_stats):
    mo.md(f"### Raw Data\n{mo_stats}")
    return


@app.cell
def _(df, mo):
    mo.ui.table(df.reset_index().tail(20), label="Recent data points")
    return


@app.cell
def _(mo):
    start_year = mo.ui.slider(
        start=2020, stop=2025, value=2024, label="Start year", step=1
    )
    resample_freq = mo.ui.dropdown(
        options={"Daily": "D", "Weekly": "W", "Monthly": "ME"},
        value="Weekly",
        label="Resample frequency",
    )
    mo.hstack([start_year, resample_freq], justify="start")
    return resample_freq, start_year


@app.cell
def _(df, np, pd, resample_freq, start_year):
    # Filter and resample
    _start = f"{start_year.value}-01-01"
    df_filtered = df.loc[_start:]

    freq = resample_freq.value
    df_rs = df_filtered.resample(freq).mean().dropna()

    # Normalize to 100 at start of window
    swe_norm = df_rs["swe"] / df_rs["swe"].iloc[0] * 100
    overall_norm = df_rs["overall"] / df_rs["overall"].iloc[0] * 100

    # Compute the gap
    gap = swe_norm - overall_norm

    # Intervention date
    intervention = pd.Timestamp("2025-12-01")

    # Stats
    _pre = df_rs[df_rs.index < intervention]
    _post = df_rs[df_rs.index >= intervention]
    corr_pre = _pre["swe"].corr(_pre["overall"]) if len(_pre) > 5 else np.nan
    corr_post = _post["swe"].corr(_post["overall"]) if len(_post) > 5 else np.nan
    return corr_post, corr_pre, df_rs, gap, intervention, overall_norm, swe_norm


@app.cell
def _(corr_post, corr_pre, gap, mo):
    mo.md(
        f"""
        ### Key Statistics

        | Metric | Value |
        |--------|-------|
        | Pre-intervention correlation (SWE vs Overall) | **{corr_pre:.3f}** |
        | Post-intervention correlation (SWE vs Overall) | **{corr_post:.3f}** |
        | Current gap (SWE − Overall, normalized) | **{gap.iloc[-1]:+.1f} pp** |
        | Max gap | **{gap.max():+.1f} pp** |
        """
    )
    return


@app.cell
def _(go, intervention, make_subplots, overall_norm, swe_norm):
    # Main comparison chart
    fig = make_subplots(rows=1, cols=1)

    fig.add_trace(
        go.Scatter(
            x=swe_norm.index,
            y=swe_norm.values,
            name="Software Engineering",
            line=dict(color="#E74C3C", width=2.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=overall_norm.index,
            y=overall_norm.values,
            name="All Job Postings",
            line=dict(color="#7F8C8D", width=2.5),
        )
    )

    # Use add_shape + add_annotation instead of add_vline to avoid
    # plotly bug with sum() on datetime values when annotation is set
    fig.add_shape(
        type="line",
        x0=intervention,
        x1=intervention,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(dash="dot", color="#E74C3C"),
        opacity=0.5,
    )
    fig.add_annotation(
        x=intervention,
        y=1,
        yref="paper",
        text="AI coding tools<br>cross threshold",
        showarrow=False,
        xanchor="left",
        yanchor="top",
        font=dict(size=10),
    )

    fig.update_layout(
        title="Software Engineering Job Postings Are Diverging Upward",
        yaxis_title="Job Postings Index (start = 100)",
        template="plotly_white",
        height=500,
        legend=dict(x=0.02, y=0.02, xanchor="left", yanchor="bottom"),
    )
    fig
    return (fig,)


@app.cell
def _(gap, go):
    # Gap chart
    fig_gap = go.Figure()
    fig_gap.add_trace(
        go.Bar(
            x=gap.index,
            y=gap.values,
            marker_color=[
                "#E74C3C" if v > 0 else "#3498DB" for v in gap.values
            ],
            opacity=0.7,
        )
    )
    fig_gap.add_hline(y=0, line_dash="solid", line_color="black", line_width=0.5)
    fig_gap.update_layout(
        title="Gap: SWE minus Overall (normalized, percentage points)",
        yaxis_title="Percentage points",
        template="plotly_white",
        height=350,
        showlegend=False,
    )
    fig_gap
    return (fig_gap,)


@app.cell
def _(mo):
    mo.md("### CausalPy: Interrupted Time Series Analysis")
    return


@app.cell
def _(df_rs, intervention, np):
    import causalpy as cp

    # Prepare data for CausalPy
    df_cp = df_rs[["swe", "overall"]].copy()

    result = cp.InterruptedTimeSeries(
        df_cp,
        treatment_time=intervention,
        formula="swe ~ 1 + overall",
        model=cp.pymc_models.LinearRegression(
            sample_kwargs={"cores": 1, "chains": 2, "draws": 2000, "tune": 1000}
        ),
    )
    return (result,)


@app.cell
def _(result):
    fig_causal, _ = result.plot()
    fig_causal.suptitle(
        "Causal Impact: SWE Job Postings\n(controlling for overall job market)",
        y=1.02,
        fontsize=14,
    )
    fig_causal.tight_layout()
    fig_causal
    return (fig_causal,)


@app.cell
def _(mo, result):
    mo.md(f"```\n{result.summary()}\n```")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Sensitivity Analysis Ideas

        The comment on LinkedIn raised good points:

        1. **"All postings" includes SWE** — the comparison group is contaminated.
           A cleaner control would exclude software-related occupations.

        2. **Negative control group** — compare against a sector with no plausible AI exposure
           (e.g., healthcare technicians, skilled trades) to validate the method.

        3. **Additional FRED series to try:**
           - `IHLIDXUSTPNURSING` (Nursing)
           - `IHLIDXUSTPMECHANIC` (Mechanics)
           - `IHLIDXUSTPBANKING` (Banking & Finance)

        These could serve as better comparison groups for a proper diff-in-diff.
        """
    )
    return


@app.cell
def _(mo):
    mo.md("---\n*Data: Indeed via FRED | Analysis: PyMC Labs*")
    return


if __name__ == "__main__":
    app.run()
