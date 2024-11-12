import os
from dotenv import load_dotenv
import lancedb

load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    DATABASE_URL = os.getenv("DATABASE_URL")

db= lancedb.connect(Config.DATABASE_URL)

config = Config()
