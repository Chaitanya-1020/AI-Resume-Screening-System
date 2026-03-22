"""
dashboard/visualization.py
───────────────────────────
All Plotly chart factories consumed by app.py.

Each function accepts a ranked ``pd.DataFrame`` (output of
``candidate_ranker.rank_candidates``) and returns a ``plotly.graph_objects.Figure``.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ── Colour palette ─────────────────────────────────────────────────────────────
_PALETTE = px.colors.sequential.Viridis
_ACCENT  = "#7C3AED"   # violet accent used in single-series charts
_BG      = "rgba(0,0,0,0)"  # transparent background for dark-mode friendliness


# ── Score bar chart ────────────────────────────────────────────────────────────

def plot_score_bar(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """
    Horizontal bar chart of candidate final scores (top-N shown).

    Parameters
    ----------
    df : pd.DataFrame
        Ranked candidates table from ``candidate_ranker``.
    top_n : int
        Maximum number of candidates to display.
    """
    subset = df.head(top_n).copy()
    subset = subset.sort_values("final_score", ascending=True)  # highest at top

    fig = px.bar(
        subset,
        x="final_score",
        y="candidate_name",
        orientation="h",
        color="final_score",
        color_continuous_scale="Viridis",
        labels={"final_score": "Match Score", "candidate_name": "Candidate"},
        title="🏆 Candidate Match Scores",
        text=subset["final_score"].apply(lambda s: f"{s:.2%}"),
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        coloraxis_showscale=False,
        plot_bgcolor=_BG,
        paper_bgcolor=_BG,
        xaxis=dict(tickformat=".0%", range=[0, 1.05]),
        yaxis_title=None,
        font=dict(size=13),
        margin=dict(l=20, r=60, t=50, b=20),
        height=max(300, 40 * len(subset) + 80),
    )
    return fig


# ── Score distribution histogram ──────────────────────────────────────────────

def plot_score_distribution(df: pd.DataFrame) -> go.Figure:
    """
    Histogram showing the distribution of match scores across all candidates.
    """
    fig = px.histogram(
        df,
        x="final_score",
        nbins=20,
        color_discrete_sequence=[_ACCENT],
        labels={"final_score": "Match Score", "count": "# Candidates"},
        title="📊 Score Distribution",
    )
    fig.update_layout(
        plot_bgcolor=_BG,
        paper_bgcolor=_BG,
        xaxis=dict(tickformat=".0%"),
        bargap=0.05,
        font=dict(size=13),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


# ── BERT vs skill score scatter ─────────────────────────────────────────────

def plot_bert_vs_skill(df: pd.DataFrame) -> go.Figure:
    """
    Scatter plot: BERT score (x) vs skill coverage (y), sized by final score.
    """
    fig = px.scatter(
        df,
        x="bert_score",
        y="skill_score",
        size="final_score",
        color="final_score",
        color_continuous_scale="Viridis",
        hover_name="candidate_name",
        hover_data={"final_score": ":.2%", "bert_score": ":.2%", "skill_score": ":.2%"},
        labels={
            "bert_score":  "BERT Similarity",
            "skill_score":  "Skill Coverage",
            "final_score":  "Match Score",
        },
        title="🔍 BERT Score vs Skill Coverage",
        size_max=30,
    )
    fig.update_layout(
        plot_bgcolor=_BG,
        paper_bgcolor=_BG,
        xaxis=dict(tickformat=".0%"),
        yaxis=dict(tickformat=".0%"),
        font=dict(size=13),
        margin=dict(l=20, r=20, t=50, b=20),
        coloraxis_showscale=False,
    )
    return fig


# ── Skill coverage donut ───────────────────────────────────────────────────────

def plot_skill_donut(df: pd.DataFrame) -> go.Figure:
    """
    Donut chart showing the share of candidates in each quality tier.
    """
    labels = df["score_label"].value_counts()
    fig = go.Figure(
        go.Pie(
            labels=labels.index,
            values=labels.values,
            hole=0.55,
            marker=dict(colors=["#22c55e", "#eab308", "#f97316", "#ef4444"]),
            textinfo="label+percent",
        )
    )
    fig.update_layout(
        title="🎯 Candidate Quality Breakdown",
        plot_bgcolor=_BG,
        paper_bgcolor=_BG,
        font=dict(size=13),
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False,
    )
    return fig


# ── Radar chart for a single candidate ───────────────────────────────────────

def plot_candidate_radar(row: pd.Series) -> go.Figure:
    """
    Radar / spider chart showing a candidate's individual score breakdown.

    Parameters
    ----------
    row : pd.Series
        A single row from the ranked candidates DataFrame.
    """
    categories   = ["BERT Score", "Skill Coverage", "Final Score"]
    values       = [row["bert_score"], row["skill_score"], row["final_score"]]
    # Close the polygon
    cats_closed  = categories + [categories[0]]
    vals_closed  = values + [values[0]]

    fig = go.Figure(
        go.Scatterpolar(
            r=vals_closed,
            theta=cats_closed,
            fill="toself",
            fillcolor=f"rgba(124, 58, 237, 0.25)",
            line=dict(color=_ACCENT, width=2),
            name=row["candidate_name"],
        )
    )
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], tickformat=".0%"),
        ),
        title=f"📋 {row['candidate_name']} – Score Breakdown",
        plot_bgcolor=_BG,
        paper_bgcolor=_BG,
        font=dict(size=12),
        margin=dict(l=40, r=40, t=60, b=40),
        showlegend=False,
    )
    return fig


# ── Top skills bar ─────────────────────────────────────────────────────────────

def plot_skill_frequency(df_records_skills: pd.Series, top_n: int = 20) -> go.Figure:
    """
    Bar chart of the most common skills across all resumes.

    Parameters
    ----------
    df_records_skills : pd.Series
        Series where each element is a comma-separated skill string
        (the ``matched_skills`` column or a pre-flattened skill list).
    top_n : int
        How many top skills to display.
    """
    all_skills: list[str] = []
    for cell in df_records_skills:
        if cell and cell != "—":
            all_skills.extend([s.strip() for s in cell.split(",") if s.strip()])

    if not all_skills:
        fig = go.Figure()
        fig.update_layout(title="No skill data available.")
        return fig

    from collections import Counter
    counts = Counter(all_skills)
    top = pd.DataFrame(counts.most_common(top_n), columns=["skill", "count"])
    top = top.sort_values("count", ascending=True)

    fig = px.bar(
        top,
        x="count",
        y="skill",
        orientation="h",
        color="count",
        color_continuous_scale="Viridis",
        labels={"count": "# Candidates", "skill": "Skill"},
        title=f"🛠️ Top {top_n} Skills Across Candidates",
        text="count",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        coloraxis_showscale=False,
        plot_bgcolor=_BG,
        paper_bgcolor=_BG,
        font=dict(size=13),
        margin=dict(l=20, r=60, t=50, b=20),
        height=max(300, 35 * len(top) + 80),
        yaxis_title=None,
    )
    return fig
