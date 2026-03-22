"""
config.py - Global configuration and constants for the AI Resume Screening System.
"""

import os

# ── File handling ────────────────────────────────────────────────────────────
SUPPORTED_FILE_TYPES: list[str] = ["pdf", "docx"]
MAX_FILE_SIZE_MB: int = 10
MAX_RESUMES: int = 20

# ── NLP ──────────────────────────────────────────────────────────────────────
SPACY_MODEL: str = "en_core_web_sm"

# ── ML / ranking ─────────────────────────────────────────────────────────────
BERT_MODEL_NAME: str = "all-MiniLM-L6-v2"

# ── Scoring thresholds ───────────────────────────────────────────────────────
SCORE_EXCELLENT: float = 0.75   # ≥ this  → "Excellent"
SCORE_GOOD: float = 0.50        # ≥ this  → "Good"
SCORE_AVERAGE: float = 0.25     # ≥ this  → "Average"
# below SCORE_AVERAGE           → "Poor"

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
DATA_DIR: str = os.path.join(BASE_DIR, "data")
SKILLS_CSV: str = os.path.join(DATA_DIR, "skills.csv")

# ── Dashboard ─────────────────────────────────────────────────────────────────
APP_TITLE: str = "AI Resume Screening System"
APP_ICON: str = "🤖"
TOP_N_CANDIDATES: int = 10          # default rows shown in ranking table
CHART_COLOR_SCALE: str = "Viridis"
