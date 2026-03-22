from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from contextlib import asynccontextmanager
from src.core.config import settings
from src.core.logger import setup_logger
from src.core.exceptions import AppException, app_exception_handler
from src.db.database import connect_to_mongo, close_mongo_connection
from src.api.routes import router
import os

logger = setup_logger("api.main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up API...")
    await connect_to_mongo()
    yield
    # Shutdown
    logger.info("Shutting down API...")
    await close_mongo_connection()

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan
)

app.add_exception_handler(AppException, app_exception_handler)

app.include_router(router, prefix=settings.API_V1_STR)

# Mount frontend
frontend_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(frontend_path):
    app.mount("/ui", StaticFiles(directory=frontend_path, html=True), name="frontend")
    
@app.get("/")
async def root():
    return RedirectResponse(url="/ui")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
