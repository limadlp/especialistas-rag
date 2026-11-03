"""Environment configuration shared by the web app and CLI."""
import os
from pathlib import Path

from dotenv import load_dotenv

SOURCE_ROOT = Path(__file__).resolve().parents[2]
ROOT = SOURCE_ROOT if (SOURCE_ROOT / "pyproject.toml").is_file() else Path.cwd()
load_dotenv(ROOT / ".env")

DATA = Path(os.getenv("TAC_DATA", ROOT / "data"))
SOURCE_BOOKS = ROOT / "source_books"
EMBEDDING_MODEL = Path(os.getenv("TAC_EMBEDDING_MODEL", DATA / "models" / "multilingual-e5-small"))
NEI_GONG_CHAT_DB = Path(os.getenv("TAC_NEI_GONG_CHAT_DB", DATA / "nei_gong" / "conversations.sqlite3"))

LMSTUDIO_URL = os.getenv("TAC_LM_URL", "http://127.0.0.1:1234").rstrip("/")
LMSTUDIO_API_TOKEN = os.getenv("LMSTUDIO_API_TOKEN", "").strip()
