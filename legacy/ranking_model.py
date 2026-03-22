"""
models/ranking_model.py
────────────────────────
TF-IDF + cosine-similarity ranking model.

This module is intentionally stateless: build a new ``RankingModel`` instance
for each screening run so there is no stale state between sessions.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE, TFIDF_STOP_WORDS

logger = logging.getLogger(__name__)


class RankingModel:
    """
    Vectorises text with TF-IDF and scores each resume against a job description
    using cosine similarity.

    Parameters
    ----------
    max_features : int
        Vocabulary size cap for TF-IDF.
    ngram_range : tuple[int, int]
        N-gram range passed to TfidfVectorizer.
    stop_words : str | None
        Stop-word language string or None.

    Example
    -------
    >>> model = RankingModel()
    >>> scores = model.fit_score(job_description, resume_texts)
    """

    def __init__(
        self,
        max_features: int = TFIDF_MAX_FEATURES,
        ngram_range: tuple[int, int] = TFIDF_NGRAM_RANGE,
        stop_words: str | None = TFIDF_STOP_WORDS,
    ) -> None:
        self._vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words=stop_words,
            sublinear_tf=True,   # log(1 + tf) dampening
        )
        self._fitted: bool = False

    # ── Core API ──────────────────────────────────────────────────────────────

    def fit_score(
        self,
        job_description: str,
        resume_texts: list[str],
    ) -> np.ndarray:
        """
        Fit the TF-IDF vocabulary on [job_description] + resumes, then
        return the cosine similarity of each resume against the JD.

        Parameters
        ----------
        job_description : str
            Cleaned job description text.
        resume_texts : list[str]
            List of cleaned resume texts.

        Returns
        -------
        np.ndarray, shape (n_resumes,)
            Similarity score in [0, 1] for each resume.

        Raises
        ------
        ValueError
            If *resume_texts* is empty.
        """
        if not resume_texts:
            raise ValueError("resume_texts must not be empty.")

        corpus = [job_description] + resume_texts
        logger.info("Fitting TF-IDF on corpus size %d.", len(corpus))

        tfidf_matrix = self._vectorizer.fit_transform(corpus)
        self._fitted = True

        jd_vector     = tfidf_matrix[0]          # first row = JD
        resume_matrix = tfidf_matrix[1:]          # remaining = resumes

        similarities: np.ndarray = cosine_similarity(jd_vector, resume_matrix).flatten()
        logger.info(
            "Scores – min: %.3f  max: %.3f  mean: %.3f",
            similarities.min(), similarities.max(), similarities.mean(),
        )
        return similarities

    def get_top_terms(self, text: str, n: int = 10) -> list[str]:
        """
        Return the top-*n* TF-IDF terms for a given text (requires model to be
        fitted first).

        Useful for debugging / explainability.
        """
        if not self._fitted:
            raise RuntimeError("Call fit_score() before get_top_terms().")

        vec = self._vectorizer.transform([text])
        feature_names = self._vectorizer.get_feature_names_out()
        sorted_indices = vec.toarray()[0].argsort()[::-1]
        return [feature_names[i] for i in sorted_indices[:n]]
