import re
import spacy
from src.core.logger import setup_logger

logger = setup_logger("nlp.preprocess")

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    logger.info("Downloading Spacy model 'en_core_web_sm'...")
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

def clean_text(text: str) -> str:
    """Preprocess text: clean characters, tokenize, remove stopwords, lemmatize."""
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = text.lower()
    
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_space]
    
    return " ".join(tokens)
