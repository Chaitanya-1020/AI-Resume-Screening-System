"""
nlp/skill_extractor.py
───────────────────────
Extracts skill keywords from text using spaCy + a reference skill list.

Strategy
--------
1. Load a curated skill list from ``data/skills.csv``.
2. Use spaCy to tokenise / lemmatise the input text.
3. Match skills using exact token n-gram matching (case-insensitive).
4. Return a deduplicated, sorted list of matched skills.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import pandas as pd
import spacy

from config import SPACY_MODEL, SKILLS_CSV

logger = logging.getLogger(__name__)


# ── Skill list loading ────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_skill_set() -> frozenset[str]:
    """
    Load the canonical skill list from CSV, returned as a lower-cased frozenset.
    Result is cached so the CSV is only read once per process.
    """
    csv_path = Path(SKILLS_CSV)
    if not csv_path.exists():
        logger.warning("skills.csv not found at %s – skill extraction will be empty.", csv_path)
        return frozenset()

    df = pd.read_csv(csv_path)
    if "skill" not in df.columns:
        raise ValueError("skills.csv must have a 'skill' column.")

    skills = {s.strip().lower() for s in df["skill"].dropna() if s.strip()}
    logger.info("Loaded %d skills from %s", len(skills), csv_path)
    return frozenset(skills)


# ── spaCy model loading ───────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_spacy_model() -> spacy.language.Language:
    """
    Load the spaCy model once and cache it.
    Falls back gracefully if the model is not installed.
    """
    try:
        nlp = spacy.load(SPACY_MODEL, disable=["parser", "ner"])
        logger.info("spaCy model '%s' loaded.", SPACY_MODEL)
        return nlp
    except OSError:
        logger.warning(
            "spaCy model '%s' not found. Run: python -m spacy download %s",
            SPACY_MODEL, SPACY_MODEL,
        )
        # Return a blank English model as fallback
        return spacy.blank("en")


# ── Core extraction logic ─────────────────────────────────────────────────────

class SkillExtractor:
    """
    Extracts skills from plain text using a curated skill list and spaCy NLP.

    Usage
    -----
    >>> extractor = SkillExtractor()
    >>> extractor.extract("Experienced in Python, machine learning and Docker.")
    ['docker', 'machine learning', 'python']
    """

    def __init__(self) -> None:
        self._nlp = _load_spacy_model()
        self._skills = _load_skill_set()

    # ── Public methods ────────────────────────────────────────────────────────

    def extract(self, text: str) -> list[str]:
        """
        Extract skills present in *text*.

        Parameters
        ----------
        text : str
            Raw or lightly cleaned text from a resume or job description.

        Returns
        -------
        list[str]
            Sorted, deduplicated list of matched skills (lowercase).
        """
        if not text or not self._skills:
            return []

        text_lower = text.lower()
        found: set[str] = set()

        # Fast path: single-token skills via exact substring match
        for skill in self._skills:
            if " " not in skill:
                # Use word-boundary aware check to avoid partial matches
                import re
                pattern = r"\b" + re.escape(skill) + r"\b"
                if re.search(pattern, text_lower):
                    found.add(skill)
            else:
                # Multi-word skill: use simple substring
                if skill in text_lower:
                    found.add(skill)

        return sorted(found)

    def extract_from_job_description(self, jd_text: str) -> list[str]:
        """
        Convenience wrapper – identical to ``extract`` but named for clarity.
        """
        return self.extract(jd_text)

    def skill_overlap(
        self,
        resume_skills: list[str],
        jd_skills: list[str],
    ) -> dict[str, list[str]]:
        """
        Compute the overlap and gaps between resume skills and JD skills.

        Returns
        -------
        dict with keys:
            ``matched``  – skills present in both
            ``missing``  – skills in JD but not in resume
            ``extra``    – skills in resume but not in JD
        """
        r = set(resume_skills)
        j = set(jd_skills)
        return {
            "matched": sorted(r & j),
            "missing": sorted(j - r),
            "extra":   sorted(r - j),
        }
