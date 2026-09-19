# test_api.py
import os
import cv2
import numpy as np
from datetime import date
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

print("--- STARTING API & SCHEMA TEST SCRIPT ---")

try:
    from app.database import Base
    from app.main import app, get_db
    from app.models import Student, Attendance
    print("Imports successful!")
except Exception as e:
    print(f"Import Error: {e}")
    exit(1)

# 1. Setup isolated test database
TEST_DB_URL = "sqlite:///./test_api.db"
engine = create_engine(TEST_DB_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base.metadata.drop_all(bind=engine)
Base.metadata.create_all(bind=engine)

# Override database dependency to route API calls to test database
def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)

# Helper functions to build sample video/image artifacts
def create_dummy_image(filename="sample_face.jpg"):
    img = np.zeros((300, 300, 3), dtype=np.uint8) + 200
    cv2.imwrite(filename, img)
    return filename

def create_dummy_video(filename="sample_cctv_feed.mp4", num_frames=15):
    height, width = 480, 640
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, 10.0, (width, height))
    for _ in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8) + 128
        out.write(frame)
    out.release()
    return filename

db = TestingSessionLocal()
img_path = create_dummy_image()
video_path = create_dummy_video()

try:
    print("1. Seeding test records...")
    student1 = Student(name="Alice Smith", visual_info_path=img_path, media_type="image")
    student2 = Student(name="Bob Johnson", visual_info_path=img_path, media_type="image")
    db.add_all([student1, student2])
    db.commit()
    db.refresh(student1)
    db.refresh(student2)

    att1 = Attendance(student_id=student1.id, date=date.today(), marked_status="Present", cctv_status="detected")
    att2 = Attendance(student_id=student2.id, date=date.today(), marked_status="Absent", cctv_status="not detected")
    db.add_all([att1, att2])
    db.commit()

    # 2. Test GET /api/dashboard/summary
    print("\n2. Testing GET /api/dashboard/summary...")
    res_summary = client.get("/api/dashboard/summary")
    assert res_summary.status_code == 200, f"Expected 200, got {res_summary.status_code}"
    summary = res_summary.json()
    print("   Response:", summary)
    assert summary["total_students"] == 2
    assert summary["marked_present"] == 1
    assert summary["cctv_detected"] == 1

    # 3. Test GET /api/attendance/daily
    print("\n3. Testing GET /api/attendance/daily...")
    res_daily = client.get("/api/attendance/daily")
    assert res_daily.status_code == 200, f"Expected 200, got {res_daily.status_code}"
    daily_logs = res_daily.json()
    print("   Response:", daily_logs)
    assert len(daily_logs) == 2

    # 4. Test POST /api/cctv/verify
    print("\n4. Testing POST /api/cctv/verify...")
    res_verify = client.post("/api/cctv/verify", json={"cctv_video_path": video_path})
    assert res_verify.status_code == 200, f"Expected 200, got {res_verify.status_code}"
    verify_res = res_verify.json()
    print("   Response:", verify_res)

    print("\n=== ALL SCHEMA AND API ENDPOINT TESTS PASSED! ===")

except Exception as e:
    print(f"\nAPI TEST FAILED: {e}\n")

finally:
    db.close()
    for path in [img_path, video_path, "test_api.db"]:
        if os.path.exists(path):
            os.remove(path)

print("--- TEST SCRIPT COMPLETED ---")