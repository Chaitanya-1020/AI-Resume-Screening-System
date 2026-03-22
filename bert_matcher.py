"""
bert_matcher.py
────────────────────────
BERT-based similarity ranking model.
"""

from __future__ import annotations

import logging

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from config import BERT_MODEL_NAME

logger = logging.getLogger(__name__)

class BERTSimilarityMatcher:
    """
    Vectorises text using BERT embeddings and scores each resume against a job description
    using cosine similarity.
    """

    def __init__(self, model_name: str = BERT_MODEL_NAME) -> None:
        logger.info("Loading BERT model: %s", model_name)
        self.model = SentenceTransformer(model_name)

    def fit_score(self, job_description: str, resume_texts: list[str]) -> np.ndarray:
        """
        Compute similarity of each resume against the JD.

        Returns
        -------
        np.ndarray, shape (n_resumes,)
            Similarity score in [0, 1] for each resume.
        """
        if not resume_texts:
            raise ValueError("resume_texts must not be empty.")

        logger.info("Encoding job description...")
        jd_embedding = self.model.encode([job_description])

        logger.info("Encoding %d resumes...", len(resume_texts))
        resume_embeddings = self.model.encode(resume_texts)

        similarities: np.ndarray = cosine_similarity(jd_embedding, resume_embeddings).flatten()
        logger.info(
            "Scores - min: %.3f  max: %.3f  mean: %.3f",
            similarities.min(), similarities.max(), similarities.mean(),
        )
        return similarities
