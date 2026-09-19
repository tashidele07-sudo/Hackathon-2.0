# test_verifier.py
import os
import cv2
import numpy as np
from datetime import date
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

print("--- STARTING TEST SCRIPT ---")

try:
    from app.database import Base
    from app.models import Student, Attendance
    from app.cctv_verifier import process_cctv_video
    print("Imports successful!")
except Exception as e:
    print(f"Import Error: {e}")
    exit(1)

# 1. Setup isolated test database
TEST_DB_URL = "sqlite:///./test_verifier.db"
engine = create_engine(TEST_DB_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base.metadata.drop_all(bind=engine)
Base.metadata.create_all(bind=engine)

def create_sample_video(filename="sample_cctv.mp4", num_frames=30):
    height, width = 480, 640
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, 10.0, (width, height))
    
    for _ in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8) + 128
        out.write(frame)
        
    out.release()
    return filename

def create_sample_image(filename="sample_face.jpg"):
    img = np.zeros((300, 300, 3), dtype=np.uint8) + 200
    cv2.imwrite(filename, img)
    return filename

# Direct Execution
db = TestingSessionLocal()
cctv_file = create_sample_video()
face_file = create_sample_image()

try:
    print("1. Seeding Test Database...")
    student = Student(
        name="Test Student",
        visual_info_path=face_file,
        media_type="image"
    )
    db.add(student)
    db.commit()
    db.refresh(student)

    print(f"2. Running CCTV Verifier on '{cctv_file}'...")
    result = process_cctv_video(cctv_video_path=cctv_file, db=db, frame_skip=5)
    print("   Pipeline Execution Summary:", result)

    print("3. Verifying Database Updates...")
    attendance = db.query(Attendance).filter(
        Attendance.student_id == student.id,
        Attendance.date == date.today()
    ).first()

    if attendance:
        print("\n=== VERIFIER TEST RESULTS ===")
        print(f"Student Name    : {student.name}")
        print(f"Date            : {attendance.date}")
        print(f"Marked Status   : {attendance.marked_status}")
        print(f"CCTV Status     : {attendance.cctv_status}")
        print(f"Source Feed     : {attendance.cctv_source_path}")
        print("=============================\n")
    else:
        print("FAILED: No attendance record was generated.")

except Exception as e:
    print(f"\nVERIFIER TEST FAILED: {e}\n")

finally:
    db.close()
    if os.path.exists(cctv_file):
        os.remove(cctv_file)
    if os.path.exists(face_file):
        os.remove(face_file)
    if os.path.exists("test_verifier.db"):
        os.remove("test_verifier.db")

print("--- TEST SCRIPT COMPLETED ---")