# 🧠 AI Resume Screening System (MLOps Edition) 🚀

An advanced, production-ready AI Resume Screening system built with a full MLOps pipeline. This project upgrades a basic TF-IDF approach into a scalable, real-world system using modern NLP, backend engineering, and deployment practices.

---

## 🚀 Key Highlights

* 🔥 Replaced TF-IDF with Sentence Transformers for semantic understanding
* ⚙️ Built production-ready REST APIs using FastAPI
* 🧠 Designed hybrid ranking system (80% semantic + 20% skill matching)
* 📊 Implemented ranking evaluation using Precision@K
* 📈 Integrated MLflow for experiment tracking and model versioning
* 🐳 Dockerized system for scalable deployment
* 🗄 MongoDB integration for persistent storage

---

## ✨ Features

* 📄 Resume Parsing (PDF/DOCX support)
* 🧠 Semantic Matching using Sentence Transformers (all-MiniLM-L6-v2)
* 🛠 Skill Extraction using spaCy NLP
* 🏆 Top-K Candidate Ranking
* 📊 Hybrid Scoring System:

  * 80% Semantic Similarity
  * 20% Skill Matching
* 📈 MLflow Tracking (Precision@K, parameters, runs)
* 🌐 FastAPI Backend (real-time API)
* 🗄 MongoDB for resume & prediction storage
* 🐳 Docker + docker-compose support

---

## 🏗 Architecture

User → FastAPI → NLP Processing → Embedding Model → Ranking Engine → MongoDB → Response

---

## 🧠 How It Works

1. User uploads resumes and job description
2. Text is parsed and preprocessed
3. Sentence Transformers generate embeddings
4. Cosine similarity is computed
5. Skills are extracted and matched
6. Final score is calculated:

Final Score = 0.8 × Semantic Similarity + 0.2 × Skill Match

7. Candidates are ranked and returned

---

## 📊 Evaluation

* Implemented **Precision@K** for ranking evaluation
* MLflow tracks:

  * Parameters
  * Metrics
  * Model runs

---

## 🛠 Tech Stack

* **Language:** Python
* **Backend:** FastAPI
* **ML/NLP:** Sentence Transformers, spaCy, Scikit-learn
* **MLOps:** MLflow, Docker
* **Database:** MongoDB
* **Tools:** Git, GitHub

---

## 📂 Project Structure

```
src/
 ├── api/        # FastAPI routes
 ├── core/       # Config, logging
 ├── db/         # MongoDB integration
 ├── models/     # ML logic + MLflow
 └── nlp/        # Parsing + preprocessing
```

---

## ⚙️ Quickstart (Docker) 🐳

```bash
docker-compose up --build
```

### Access:

* FastAPI Docs → http://localhost:8000/docs
* MLflow UI → http://localhost:5000

---

## 💻 Local Setup

```bash
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Run services:

```bash
mlflow server --host 127.0.0.1 --port 5000
uvicorn src.api.main:app --reload
```

---

## 📡 API Usage

### Health Check

```
GET /api/v1/health
```

### Resume Screening

```
POST /api/v1/predict
```

**Input:**

* job_description (text)
* multiple resume files

**Output:**

```json
{
  "ranked_candidates": [
    {
      "name": "resume1.pdf",
      "similarity_score": 87.5,
      "skills_matched": ["python", "ml"],
      "rank": 1
    }
  ]
}
```

---

## ☁️ Deployment (Render)

* Deploy FastAPI backend on Render
* Use MongoDB Atlas (free tier)
* Configure environment variables

---

## 🔄 Project Evolution

Initially built using TF-IDF and cosine similarity.
Upgraded to a production-ready MLOps system with:

* Sentence Transformers
* FastAPI
* MLflow
* MongoDB
* Docker

---

## 👨‍💻 Author

**Chaitanya Mule**
