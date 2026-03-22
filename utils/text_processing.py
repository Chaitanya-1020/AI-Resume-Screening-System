"""
utils/text_processing.py
─────────────────────
Combined text pre-processing and skill extraction utilities.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import pandas as pd
import spacy

from config import SPACY_MODEL, SKILLS_CSV

logger = logging.getLogger(__name__)

# ── Cleaning Helpers ──────────────────────────────────────────────────────────

_WHITESPACE_RE = re.compile(r"\s+")
_NON_ASCII_RE  = re.compile(r"[^\x00-\x7F]+")
_URL_RE        = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_RE      = re.compile(r"\S+@\S+\.\S+")
_PHONE_RE      = re.compile(r"(\+?\d[\d\s\-().]{7,}\d)")

def _normalise_unicode(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")

def remove_urls(text: str) -> str:
    return _URL_RE.sub(" ", text)

def remove_emails(text: str) -> str:
    return _EMAIL_RE.sub(" ", text)

def remove_phone_numbers(text: str) -> str:
    return _PHONE_RE.sub(" ", text)

def collapse_whitespace(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()

def clean_text(
    text: str,
    *,
    remove_pii: bool = False,
    lowercase: bool = True,
) -> str:
    """Clean and normalise raw resume / JD text."""
    if not isinstance(text, str):
        return ""

    text = _normalise_unicode(text)

    if remove_pii:
        text = remove_urls(text)
        text = remove_emails(text)
        text = remove_phone_numbers(text)

    text = re.sub(r"[^\w\s\-]", " ", text)
    text = collapse_whitespace(text)

    if lowercase:
        text = text.lower()

    return text

def truncate_text(text: str, max_chars: int = 10_000) -> str:
    if len(text) <= max_chars:
        return text
    cutoff = text.rfind(" ", 0, max_chars)
    return text[:cutoff] if cutoff > 0 else text[:max_chars]


# ── Skill Extractor ───────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_skill_set() -> frozenset[str]:
    csv_path = Path(SKILLS_CSV)
    if not csv_path.exists():
        logger.warning("skills.csv not found at %s – skill extraction will be empty.", csv_path)
        return frozenset()

    df = pd.read_csv(csv_path)
    if "skill" not in df.columns:
        raise ValueError("skills.csv must have a 'skill' column.")

    skills = {s.strip().lower() for s in df["skill"].dropna() if s.strip()}
    return frozenset(skills)

@lru_cache(maxsize=1)
def _load_spacy_model() -> spacy.language.Language:
    try:
        nlp = spacy.load(SPACY_MODEL, disable=["parser", "ner"])
        return nlp
    except OSError:
        logger.warning("spaCy model '%s' not found. Returning blank en.", SPACY_MODEL)
        return spacy.blank("en")

class SkillExtractor:
    def __init__(self) -> None:
        self._nlp = _load_spacy_model()
        self._skills = _load_skill_set()

    def extract(self, text: str) -> list[str]:
        if not text or not self._skills:
            return []

        text_lower = text.lower()
        found: set[str] = set()

        for skill in self._skills:
            if " " not in skill:
                pattern = r"\b" + re.escape(skill) + r"\b"
                if re.search(pattern, text_lower):
                    found.add(skill)
            else:
                if skill in text_lower:
                    found.add(skill)

        return sorted(found)

    def extract_from_job_description(self, jd_text: str) -> list[str]:
        return self.extract(jd_text)

    def skill_overlap(self, resume_skills: list[str], jd_skills: list[str]) -> dict[str, list[str]]:
        r = set(resume_skills)
        j = set(jd_skills)
        return {
            "matched": sorted(r & j),
            "missing": sorted(j - r),
            "extra": sorted(r - j),
        }
