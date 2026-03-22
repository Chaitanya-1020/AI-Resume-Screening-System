import io
import pdfplumber
import docx
from src.core.logger import setup_logger

logger = setup_logger("nlp.resume_parser")

def extract_text_from_pdf(file_bytes: bytes) -> str:
    text = ""
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
    except Exception as e:
        logger.error(f"Error reading PDF: {e}")
    return text

def extract_text_from_docx(file_bytes: bytes) -> str:
    text = ""
    try:
        doc = docx.Document(io.BytesIO(file_bytes))
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        logger.error(f"Error reading DOCX: {e}")
    return text

def parse_resume(filename: str, file_bytes: bytes) -> str:
    if filename.lower().endswith('.pdf'):
        return extract_text_from_pdf(file_bytes)
    elif filename.lower().endswith('.docx') or filename.lower().endswith('.doc'):
        return extract_text_from_docx(file_bytes)
    elif filename.lower().endswith('.txt'):
        return file_bytes.decode('utf-8', errors='ignore')
    else:
        logger.warning(f"Unsupported file format for {filename}")
        return ""
