# AI Resume Screening System (MLOps Edition) 🚀

An advanced, production-ready AI Resume Screening API built with FastAPI, Sentence-Transformers, MongoDB, MLflow, and Docker. This project upgrades a basic TF-IDF matching script into a robust, scalable MLOps pipeline.

## Features ✨
- **Accurate NLP Matching**: Uses `SentenceTransformers` (`all-MiniLM-L6-v2`) for deep semantic text embeddings instead of basic TF-IDF.
- **Robust Preprocessing**: Lemmatization, tokenization, stopword removal using `Spacy`.
- **Automated Skill Extraction**: Identifies technical skills from resumes automatically.
- **MLflow Tracking**: Tracks model training metrics (Precision/Recall proxies), hyperparameters, and model versions.
- **FastAPI Backend**: Asynchronous endpoints for health checking and fast predictions.
- **MongoDB Storage**: Stores parsed resumes and prediction histories for auditing.
- **Dockerized**: Fully containerized with `docker-compose` for instantaneous local setup.

## Project Structure 📂
```
.
├── src/
│   ├── api/          # FastAPI routes and entrypoint
│   ├── core/         # Global config, logging, and exception handling
│   ├── db/           # MongoDB connection and Pydantic models
│   ├── models/       # MLflow training script and Sentence-Transformer prediction logic
│   └── nlp/          # Text preprocessing, skill extraction, and PDF/DOCX parsing
├── venv/             # Python Virtual Environment
├── Dockerfile        # Docker container definition
├── docker-compose.yml# Multi-container orchestration (API + MongoDB + MLflow)
├── requirements.txt  # Python Dependencies
└── README.md         # Documentation
```

## Quickstart (Docker - Recommended) 🐳

The easiest way to run the entire backend, MongoDB, and MLflow UI is via Docker Compose.

1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop).
2. Run the application:
   ```bash
   docker-compose up --build
   ```
3. Access the services:
   - **FastAPI Docs (Swagger UI)**: http://localhost:8000/docs
   - **MLflow Tracking UI**: http://localhost:5000

## Manual Setup (Local Python) 💻

1. **Install Dependencies**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```
2. **Setup MongoDB**:
   Download and install [MongoDB Community Server](https://www.mongodb.com/try/download/community) and ensure it runs on `localhost:27017`.
   
3. **Run MLflow**:
   In a new terminal:
   ```bash
   mlflow server --host 127.0.0.1 --port 5000
   ```

4. **Run FastAPI Server**:
   ```bash
   uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
   ```

## Cloud Deployment Guide (Render - Free Tier) ☁️

1. Create a free account on [Render](https://render.com) and link your GitHub repository.
2. Create a Free **MongoDB Atlas** cluster and get your connection string (e.g., `mongodb+srv://user:pass@cluster.mongodb.net/`).
3. In Render, select **New > Web Service** and choose your repository.
4. Settings:
   - **Build Command**: `pip install -r requirements.txt && python -m spacy download en_core_web_sm`
   - **Start Command**: `uvicorn src.api.main:app --host 0.0.0.0 --port $PORT`
5. Environment Variables:
   - Add `MONGO_URI` and paste your MongoDB Atlas string.
   - Add `MONGO_DB_NAME` = `ai_resume_screening`
   - Add `PYTHON_VERSION` = `3.10`
6. Click **Deploy Web Service**!

## API Usage 📡

**1. Health Check**
```http
GET /api/v1/health
```

**2. Predict / Screen Resumes**
```http
POST /api/v1/predict
Content-Type: multipart/form-data

form-data:
- job_description: (text) "Looking for a Python backend engineer..."
- files: (File) Resume1.pdf
- files: (File) Resume2.docx
```
*Returns ranked JSON output containing similarity scores and matched skills.*
