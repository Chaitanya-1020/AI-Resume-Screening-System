import motor.motor_asyncio
from src.core.config import settings
from src.core.logger import setup_logger

logger = setup_logger("database")

class Database:
    client: motor.motor_asyncio.AsyncIOMotorClient = None
    db = None

db_config = Database()

async def connect_to_mongo():
    logger.info("Connecting to MongoDB...")
    db_config.client = motor.motor_asyncio.AsyncIOMotorClient(settings.MONGO_URI)
    db_config.db = db_config.client[settings.MONGO_DB_NAME]
    logger.info("Connected to MongoDB!")

async def close_mongo_connection():
    logger.info("Closing MongoDB connection...")
    if db_config.client:
        db_config.client.close()
        logger.info("MongoDB connection closed.")

def get_database():
    return db_config.db
