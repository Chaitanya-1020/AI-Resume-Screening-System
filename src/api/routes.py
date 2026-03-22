from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from typing import List
from src.core.logger import setup_logger
from src.nlp.resume_parser import parse_resume
from src.nlp.preprocess import clean_text
from src.nlp.skill_extractor import extract_skills
from src.models.predict import get_embedding, compute_similarity
from src.db.database import get_database
from src.db.models import ResumeRecord, PredictionRecord

logger = setup_logger("api.routes")
router = APIRouter()

@router.get("/health")
async def health_check():
    return {"status": "ok", "message": "MLOps Resume API is running!"}

@router.post("/predict")
async def predict(job_description: str = Form(...), top_k: int = Form(5), files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No resume files uploaded.")
    if not job_description.strip():
        raise HTTPException(status_code=400, detail="Job description cannot be empty.")

    logger.info(f"Received prediction request with {len(files)} resumes. Top K: {top_k}")
    
    db = get_database()
    
    # Process Job Description
    clean_jd = clean_text(job_description)
    jd_embedding = get_embedding(clean_jd)
    jd_skills = extract_skills(job_description)
    jd_skills_set = set(jd_skills)
    
    candidates = []
    
    for file in files:
        if not file.filename.lower().endswith(('.pdf', '.docx', '.txt')):
            logger.warning(f"Invalid file type for {file.filename}")
            continue
            
        content = await file.read()
        text = parse_resume(file.filename, content)
        
        if not text.strip():
            logger.warning(f"Could not extract text from {file.filename}")
            continue
            
        clean_res = clean_text(text)
        res_embedding = get_embedding(clean_res)
        
        sim = compute_similarity(jd_embedding, res_embedding)
        semantic_score = float(max(0, sim) * 100)
        
        resume_skills = extract_skills(text)
        resume_skills_set = set(resume_skills)
        matched_skills = list(jd_skills_set & resume_skills_set)
        
        if jd_skills_set:
            skill_score = (len(matched_skills) / len(jd_skills_set)) * 100
        else:
            skill_score = 100.0  # If no skills to match, assume perfect match
            
        final_score = round(0.8 * semantic_score + 0.2 * skill_score, 2)
        
        # Save resume record
        resume_record = ResumeRecord(
            filename=file.filename,
            text_content=text,
            extracted_skills=resume_skills
        )
        if db is not None:
            await db["resumes"].insert_one(resume_record.model_dump(by_alias=True, exclude={"id"}))
            logger.info(f"Saved resume {file.filename} to DB.")
        
        candidates.append({
            "name": file.filename,
            "similarity_score": final_score,
            "semantic_score": round(semantic_score, 2),
            "skill_score": round(skill_score, 2),
            "skills_matched": matched_skills,
        })

    # Sort results
    candidates = sorted(candidates, key=lambda x: x["similarity_score"], reverse=True)
    
    # Select top K and assign ranks
    ranked_candidates = []
    for i, candidate in enumerate(candidates[:top_k]):
        candidate["rank"] = i + 1
        ranked_candidates.append(candidate)
    
    # Save prediction history
    if db is not None:
        prediction_record = PredictionRecord(
            job_description=job_description,
            candidates_ranked=ranked_candidates
        )
        await db["predictions"].insert_one(prediction_record.model_dump(by_alias=True, exclude={"id"}))
        logger.info("Saved prediction history to DB.")
    
    return {
        "job_description_length": len(job_description),
        "total_resumes_processed": len(candidates),
        "top_k_returned": len(ranked_candidates),
        "ranked_candidates": ranked_candidates
    }
