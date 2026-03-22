from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
from datetime import datetime

class ResumeRecord(BaseModel):
    id: Optional[str] = Field(alias="_id", default=None)
    filename: str
    text_content: str
    extracted_skills: List[str]
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    model_config = ConfigDict(populate_by_name=True)

class PredictionRecord(BaseModel):
    id: Optional[str] = Field(alias="_id", default=None)
    job_description: str
    candidates_ranked: List[dict]
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    model_config = ConfigDict(populate_by_name=True)
