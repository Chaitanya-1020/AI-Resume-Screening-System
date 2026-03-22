import re

COMMON_SKILLS = [
    "python", "java", "c++", "c#", "javascript", "typescript", "golang",
    "machine learning", "deep learning", "nlp", "computer vision", "mlops",
    "docker", "kubernetes", "aws", "gcp", "azure", "sql", "nosql", "mongodb",
    "fastapi", "flask", "django", "react", "angular", "vue", "pytorch",
    "tensorflow", "keras", "scikit-learn", "pandas", "numpy", "spacy",
    "git", "linux", "agile", "scrum", "bert", "transformers"
]

def extract_skills(text: str) -> list[str]:
    """Extract predefined skills from text."""
    text_lower = text.lower()
    found_skills = set()
    for skill in COMMON_SKILLS:
        # Match word boundaries for skills
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            found_skills.add(skill)
    return list(found_skills)
