import os
import mlflow
from src.models.predict import compute_similarity, get_embedding
from src.nlp.preprocess import clean_text
from src.nlp.skill_extractor import extract_skills
from src.core.config import settings
from src.core.logger import setup_logger

logger = setup_logger("models.train")

def run_experiment():
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment("Resume_Screening_Evaluation")
    
    logger.info("Starting MLflow experiment...")
    
    with mlflow.start_run():
        mlflow.log_param("model_name", settings.EMBEDDING_MODEL_NAME)
        mlflow.log_param("preprocessing", "lowercase, remove stopwords, lemmatize")
        
        # Test Data
        job_description = "We need a machine learning engineer with Python, PyTorch, MLflow and NLP experience."
        resumes = [
            {"id": "r1", "text": "Experienced ML Engineer. Skilled in Python, Deep Learning, NLP, MLflow and PyTorch. Built sentiment analysis models.", "ground_truth": 1},
            {"id": "r2", "text": "Frontend developer. React, JavaScript, HTML, CSS.", "ground_truth": 0},
            {"id": "r3", "text": "Data Scientist with experience in Python, Pandas, and Scikit-Learn.", "ground_truth": 0},
            {"id": "r4", "text": "Senior NLP Engineer, PyTorch expert, Python developer, familiar with MLflow.", "ground_truth": 1},
            {"id": "r5", "text": "Backend developer with Java and Spring Boot.", "ground_truth": 0}
        ]
        
        K = 2 # Top K evaluation
        
        clean_jd = clean_text(job_description)
        jd_embedding = get_embedding(clean_jd)
        jd_skills_set = set(extract_skills(job_description))
        
        candidates = []
        
        for res in resumes:
            clean_res = clean_text(res["text"])
            res_embedding = get_embedding(clean_res)
            
            sim = compute_similarity(jd_embedding, res_embedding)
            semantic_score = float(max(0, sim) * 100)
            
            resume_skills_set = set(extract_skills(res["text"]))
            matched_skills = list(jd_skills_set & resume_skills_set)
            
            if jd_skills_set:
                skill_score = (len(matched_skills) / len(jd_skills_set)) * 100
            else:
                skill_score = 100.0
                
            final_score = round(0.8 * semantic_score + 0.2 * skill_score, 2)
            
            candidates.append({
                "id": res["id"],
                "score": final_score,
                "ground_truth": res["ground_truth"]
            })
            
        candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
        top_k_candidates = candidates[:K]
        
        relevant_in_top_k = sum(1 for c in top_k_candidates if c["ground_truth"] == 1)
        precision_at_k = relevant_in_top_k / K
        
        logger.info(f"Top-{K} Candidates: {[c['id'] for c in top_k_candidates]}")
        logger.info(f"Precision@{K}: {precision_at_k:.2f}")
        
        mlflow.log_metric(f"precision_at_{K}", precision_at_k)
        mlflow.log_param("top_k", K)
        mlflow.log_param("dataset_size", len(resumes))
        
        logger.info("Experiment completed and logged to MLflow.")

if __name__ == "__main__":
    run_experiment()
