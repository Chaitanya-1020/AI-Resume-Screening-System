from sentence_transformers import SentenceTransformer, util
from src.core.config import settings
from src.core.logger import setup_logger

logger = setup_logger("models.predict")

logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL_NAME}")
try:
    model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None

def get_embedding(text: str):
    if not model:
        raise ValueError("Model not loaded.")
    return model.encode(text, convert_to_tensor=True)

def compute_similarity(job_desc_embedding, resume_embedding) -> float:
    cosine_scores = util.cos_sim(job_desc_embedding, resume_embedding)
    return cosine_scores.item()
