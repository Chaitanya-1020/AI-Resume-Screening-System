"""
app.py – AI Resume Screening System  (v3.0)
────────────────────────────────────────────
New in v3:
  • BERT-based Candidate Matching
  • Interactive Dashboard and Score Refinements

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import streamlit as st

from config import APP_ICON, APP_TITLE, MAX_FILE_SIZE_MB, MAX_RESUMES
from dashboard.visualization import (
    plot_bert_vs_skill,
    plot_candidate_radar,
    plot_score_bar,
    plot_score_distribution,
    plot_skill_donut,
    plot_skill_frequency,
)
from ranking import rank_candidates
from resume_parser import process_resumes

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Space+Mono:wght@700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4f46e5 100%);
        padding: 2rem 2.5rem 1.5rem; border-radius: 16px; margin-bottom: 1.5rem;
        box-shadow: 0 4px 24px rgba(79,70,229,0.3);
    }
    .main-header h1 {
        font-family: 'Space Mono', monospace; color: #ffffff; font-size: 2rem;
        margin: 0; letter-spacing: -0.5px;
    }
    .main-header p { color: #c7d2fe; margin: 0.4rem 0 0; font-size: 1rem; }
    .main-header .badge {
        display: inline-block; background: #4f46e5; color: #fff; font-size: 0.7rem;
        padding: 2px 10px; border-radius: 20px; margin-left: 10px;
        vertical-align: middle; font-weight: 600; letter-spacing: 0.5px;
    }
    .metric-card {
        background: linear-gradient(135deg, #312e81, #1e1b4b);
        border: 1px solid #4f46e5; border-radius: 12px; padding: 1.2rem;
        text-align: center; color: #fff; box-shadow: 0 2px 12px rgba(79,70,229,0.2);
    }
    .metric-card .value {
        font-family: 'Space Mono', monospace; font-size: 2rem;
        font-weight: 700; color: #a5b4fc;
    }
    .metric-card .label { font-size: 0.8rem; color: #c7d2fe; margin-top: 0.2rem; }
    .section-title {
        font-size: 1.2rem; font-weight: 700; color: #4f46e5;
        border-left: 4px solid #4f46e5; padding-left: 0.6rem;
        margin: 1.5rem 0 0.8rem;
    }
    .skill-pill {
        display: inline-block; background: #e0e7ff; color: #3730a3;
        border-radius: 20px; padding: 2px 10px; font-size: 0.78rem;
        margin: 2px; font-weight: 600;
    }
    .skill-pill.missing { background: #fee2e2; color: #991b1b; }
    [data-testid="stSidebar"] { background: #0f0e2a; }
    [data-testid="stSidebar"] * { color: #c7d2fe !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session state defaults ────────────────────────────────────────────────────
_DEFAULTS: dict = {
    "ranked_df":         None,
    "jd_skill_count":    0,
    "processing_errors": [],
    "jd_text":           "",
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="main-header">
        <h1>{APP_ICON} {APP_TITLE} <span class="badge">v3.0</span></h1>
        <p>Upload resumes &middot; Enter a job description &middot;
           Rank candidates using advanced BERT match</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=60)
    st.markdown("### Configuration")
    st.caption(
        f"Upload up to **{MAX_RESUMES}** resumes (PDF or DOCX, max {MAX_FILE_SIZE_MB} MB each)."
    )

    uploaded_files = st.file_uploader(
        "Upload Resumes",
        type=["pdf", "docx"],
        accept_multiple_files=True,
    )

    st.markdown("---")
    jd_input = st.text_area(
        "Job Description",
        height=260,
        placeholder=(
            "Paste the full job description here...\n\n"
            "e.g. We are looking for a Senior Python Developer with "
            "experience in machine learning, scikit-learn, Docker, "
            "and REST APIs..."
        ),
    )

    st.markdown("---")
    run_btn   = st.button("Screen Candidates", type="primary", use_container_width=True)
    clear_btn = st.button("Clear Results", use_container_width=True)

    if clear_btn:
        for _k in _DEFAULTS:
            st.session_state[_k] = _DEFAULTS[_k]
        st.rerun()

    st.markdown("---")
    st.caption("**AI Resume Screening v3.0**\nPython / spaCy / scikit-learn / Streamlit")


# ══════════════════════════════════════════════════════════════════════════════
# HELPER RENDERERS
# ══════════════════════════════════════════════════════════════════════════════

def _render_metric_cards(df: pd.DataFrame) -> None:
    """Top-level KPI row."""
    c1, c2, c3, c4 = st.columns(4)
    cards = [
        (len(df),                           "Candidates Processed"),
        (f"{df['final_score'].max():.0%}",  "Best Match Score"),
        (f"{df['final_score'].mean():.0%}", "Average Match Score"),
        (int((df["score_label"].str.startswith("🟢")).sum()), "Excellent Matches"),
    ]
    for col, (val, label) in zip([c1, c2, c3, c4], cards):
        with col:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="value">{val}</div>'
                f'<div class="label">{label}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


def _render_ranking_table(df: pd.DataFrame) -> None:
    """Ranked candidate table with CSV export."""
    st.markdown('<div class="section-title">Candidate Rankings</div>', unsafe_allow_html=True)
    disp = df[[
        "rank", "candidate_name", "score_label", "final_score",
        "bert_score", "skill_score", "jd_skill_coverage_%", "total_skills",
    ]].copy()
    for col in ("final_score", "bert_score", "skill_score"):
        disp[col] = disp[col].apply(lambda x: f"{x:.2%}")
    disp.columns = [
        "Rank", "Candidate", "Tier", "Final Score",
        "BERT Score", "Skill Score", "JD Skill Coverage %", "Total Skills",
    ]
    st.dataframe(disp, use_container_width=True, hide_index=True)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Full Results (CSV)", csv,
        file_name="ranked_candidates.csv", mime="text/csv",
    )


def _render_charts(df: pd.DataFrame) -> None:
    """All Plotly visualisation panels."""
    st.markdown('<div class="section-title">Visual Analytics</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([2, 1])
    with c1:
        st.plotly_chart(plot_score_bar(df), use_container_width=True)
    with c2:
        st.plotly_chart(plot_skill_donut(df), use_container_width=True)
    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(plot_score_distribution(df), use_container_width=True)
    with c4:
        st.plotly_chart(plot_bert_vs_skill(df), use_container_width=True)
    st.plotly_chart(plot_skill_frequency(df["matched_skills"]), use_container_width=True)


def _render_deep_dive(df: pd.DataFrame) -> None:
    """Per-candidate radar chart and skills breakdown."""
    st.markdown('<div class="section-title">Candidate Deep Dive</div>', unsafe_allow_html=True)
    selected = st.selectbox("Select a candidate to inspect", df["candidate_name"].tolist())
    row = df[df["candidate_name"] == selected].iloc[0]
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plot_candidate_radar(row), use_container_width=True)
    with c2:
        st.markdown(f"**Rank:** {int(row['rank'])} of {len(df)}")
        st.markdown(f"**Final Score:** {row['final_score']:.2%}")
        st.markdown(f"**Tier:** {row['score_label']}")
        st.markdown(f"**JD Skill Coverage:** {row['jd_skill_coverage_%']:.1f}%")
        st.markdown(f"**Total Skills Detected:** {int(row['total_skills'])}")
        st.markdown("**Matched Skills:**")
        if row["matched_skills"] != "-":
            pills = "".join(
                f'<span class="skill-pill">{s.strip()}</span>'
                for s in str(row["matched_skills"]).split(",")
                if s.strip()
            )
            st.markdown(pills, unsafe_allow_html=True)
        else:
            st.caption("No matched skills.")
        st.markdown("**Missing Skills (from JD):**")
        if row["missing_skills"] != "-":
            pills = "".join(
                f'<span class="skill-pill missing">{s.strip()}</span>'
                for s in str(row["missing_skills"]).split(",")
                if s.strip()
            )
            st.markdown(pills, unsafe_allow_html=True)
        else:
            st.caption("All JD skills matched!")


# ══════════════════════════════════════════════════════════════════════════════
# PROCESSING TRIGGER
# ══════════════════════════════════════════════════════════════════════════════
if run_btn:
    if not uploaded_files:
        st.warning("Please upload at least one resume.")
    elif not jd_input.strip():
        st.warning("Please enter a job description.")
    elif len(uploaded_files) > MAX_RESUMES:
        st.error(f"Maximum {MAX_RESUMES} resumes allowed. You uploaded {len(uploaded_files)}.")
    else:
        with st.spinner("Processing resumes and ranking candidates..."):
            file_tuples = [(f.read(), f.name) for f in uploaded_files]
            records, errors = process_resumes(file_tuples)
            st.session_state.processing_errors = errors

            if not records:
                st.error("No valid resumes could be processed. Check file formats.")
            else:
                try:
                    df_result, jd_skill_count = rank_candidates(records, jd_input)

                    st.session_state.ranked_df      = df_result
                    st.session_state.jd_skill_count = jd_skill_count
                    st.session_state.jd_text        = jd_input

                    st.success(
                        f"Successfully screened **{len(records)}** candidate(s)!"
                    )
                except ValueError as exc:
                    st.error(f"Ranking failed: {exc}")


# ── Show parsing errors ───────────────────────────────────────────────────────
if st.session_state.processing_errors:
    with st.expander("Parsing Errors", expanded=False):
        for err in st.session_state.processing_errors:
            st.markdown(err)


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
_df: pd.DataFrame | None         = st.session_state.ranked_df

if _df is not None and not _df.empty:
    _render_metric_cards(_df)
    st.divider()
    _render_ranking_table(_df)
    st.divider()
    _render_charts(_df)
    st.divider()
    _render_deep_dive(_df)
    st.divider()

else:
    st.markdown(
        """
        <div style="text-align:center; padding: 4rem 2rem; opacity: 0.55;">
            <p style="font-size:4rem; margin:0;">📄</p>
            <p style="font-size:1.4rem; font-weight:600; margin:0.5rem 0;">No results yet</p>
            <p style="font-size:1rem;">
                Upload resumes and enter a job description in the sidebar,
                then click <strong>Screen Candidates</strong>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
