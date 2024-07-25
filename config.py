from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    api_key: str

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()