import os
import shutil
import datetime
from app.database import SessionLocal, engine, Base
from app.models import Student, Attendance
from app.config import STORAGE_DIR
from app.face_engine import extract_face_encoding

# Ensure database tables exist
Base.metadata.create_all(bind=engine)

# Define 4 student records matching your image files
STUDENTS_TO_REGISTER = [
    {"name": "Alice Smith", "lcid": "101", "image_file": "student1.jpg"},
    {"name": "Bob Jones", "lcid": "102", "image_file": "student2.jpg"},
    {"name": "Charlie Brown", "lcid": "103", "image_file": "student3.jpg"},
    {"name": "Diana Prince", "lcid": "104", "image_file": "student4.jpg"},
]

def seed_database():
    db = SessionLocal()
    today_str = datetime.date.today().strftime("%Y-%m-%d")

    for data in STUDENTS_TO_REGISTER:
        source_img = data["image_file"]
        lcid = data["lcid"]

        if not os.path.exists(source_img):
            print(f"[SKIP] Source image '{source_img}' not found.")
            continue

        # 1. Check if student already exists
        existing = db.query(Student).filter(Student.lcid == lcid).first()
        if existing:
            print(f"[EXISTS] Student Roll {roll} ({data['name']}) already in database.")
            continue

        # 2. Copy image to storage folder
        ext = source_img.split(".")[-1]
        stored_filename = f"{lcid}.{ext}"
        destination_path = os.path.join(STORAGE_DIR, stored_filename)
        shutil.copy(source_img, destination_path)

        # 3. Extract 128D facial encoding vector
        try:
            encoding_json = extract_face_encoding(destination_path)
        except Exception as e:
            print(f"[ERROR] Could not extract face from {source_img}: {e}")
            os.remove(destination_path)
            continue

        # 4. Save Student profile to Database
        student = Student(
            name=data["name"],
            lcid=lcid,
            face_image_path=stored_filename,
            face_encoding=encoding_json
        )
        db.add(student)
        db.commit()
        db.refresh(student)

        # 5. Create initial daily Attendance entry (PENDING status)
        attendance = Attendance(
            student_id=student.id,
            date=today_str,
            verification_status="PENDING"
        )
        db.add(attendance)
        db.commit()

        print(f"[SUCCESS] Registered {data['name']} (Roll: {roll}) with initial PENDING attendance.")

    db.close()

if __name__ == "__main__":
    seed_database()