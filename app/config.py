import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
STORAGE_DIR = BASE_DIR / "storage" / "student_faces"
DATABASE_URL = f"sqlite:///{BASE_DIR}/vision_attend.db"

# Ensure visual storage directory exists
os.makedirs(STORAGE_DIR, exist_ok=True)