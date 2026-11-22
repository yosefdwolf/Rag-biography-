import os

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL") or "gpt-4.1-mini"
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL") or "text-embedding-3-small"

# Railway hands out postgresql:// URLs; SQLAlchemy needs the psycopg (v3) driver named.
_db = os.getenv("DATABASE_URL") or "sqlite:///./local.db"
if _db.startswith("postgres://"):
    _db = "postgresql://" + _db[len("postgres://"):]
if _db.startswith("postgresql://"):
    _db = "postgresql+psycopg://" + _db[len("postgresql://"):]
DATABASE_URL = _db

TOP_K = int(os.getenv("TOP_K") or 6)
# A chunk is kept only if it clears both: an absolute floor, and standing out from the rest of the
# corpus for this question (z-score = how many standard deviations above the average chunk's score).
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY") or 0.20)
MIN_Z = float(os.getenv("MIN_Z") or 2.0)
CHUNK_WORDS = int(os.getenv("CHUNK_WORDS") or 350)
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP") or 60)

# Required: uploads, runs and questions need this password. Unset = they are refused.
APP_PASSWORD = os.getenv("APP_PASSWORD", "")
