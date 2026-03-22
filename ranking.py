"""
ranking.py
──────────────────────────────
Ranks candidates by combining BERT cosine-similarity score and Skill-overlap.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from bert_matcher import BERTSimilarityMatcher
from config import SCORE_AVERAGE, SCORE_EXCELLENT, SCORE_GOOD
from resume_parser import ResumeRecord
from utils.text_processing import SkillExtractor, clean_text

logger = logging.getLogger(__name__)

_extractor = SkillExtractor()
_BERT_WEIGHT = 0.80
_SKILL_WEIGHT = 0.20

# ── Helpers ───────────────────────────────────────────────────────────────────
def _score_label(score: float) -> str:
    if score >= SCORE_EXCELLENT:
        return "🟢 Excellent"
    elif score >= SCORE_GOOD:
        return "🟡 Good"
    elif score >= SCORE_AVERAGE:
        return "🟠 Average"
    return "🔴 Poor"

def _skill_overlap_ratio(resume_skills: list[str], jd_skills: list[str]) -> float:
    if not jd_skills:
        return 0.0
    matched = set(resume_skills) & set(jd_skills)
    return len(matched) / len(jd_skills)

# ── Public API ────────────────────────────────────────────────────────────────
def rank_candidates(records: list[ResumeRecord], job_description: str) -> tuple[pd.DataFrame, int]:
    """
    Score and rank all valid candidates against the job description using BERT embeddings.
    """
    if not records:
        raise ValueError("No valid resumes to rank.")
    if not job_description.strip():
        raise ValueError("Job description cannot be empty.")

    clean_jd = clean_text(job_description, remove_pii=False, lowercase=True)
    jd_skills = _extractor.extract_from_job_description(clean_jd)
    logger.info("JD skills detected: %s", jd_skills)

    # BERT similarities
    matcher = BERTSimilarityMatcher()
    resume_texts = [r.clean_text for r in records]
    bert_scores = matcher.fit_score(clean_jd, resume_texts)

    # Skill overlap scores
    skill_scores = np.array([_skill_overlap_ratio(r.skills, jd_skills) for r in records])

    # Weighted final score
    final_scores = _BERT_WEIGHT * bert_scores + _SKILL_WEIGHT * skill_scores

    # Build result rows
    rows = []
    for i, record in enumerate(records):
        overlap = _extractor.skill_overlap(record.skills, jd_skills)
        rows.append({
            "candidate_name": record.candidate_name,
            "filename": record.filename,
            "final_score": round(float(final_scores[i]), 4),
            "bert_score": round(float(bert_scores[i]), 4),
            "skill_score": round(float(skill_scores[i]), 4),
            "matched_skills": ", ".join(overlap["matched"]) or "—",
            "missing_skills": ", ".join(overlap["missing"]) or "—",
            "total_skills": len(record.skills),
            "jd_skill_coverage_%": round(float(skill_scores[i]) * 100, 1),
        })

    df = pd.DataFrame(rows)
    df.sort_values("final_score", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    df["score_label"] = df["final_score"].apply(_score_label)

    return df, len(jd_skills)
