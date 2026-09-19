from datetime import date
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.models import Base, Student, Attendance

# Step 1: Initialize temporary SQLite database
DATABASE_URL = "sqlite:///./test_database.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Step 2: Generate tables based on models.py
Base.metadata.drop_all(bind=engine)
Base.metadata.create_all(bind=engine)

def verify_database_schema():
    db = TestingSessionLocal()
    try:
        # Step 3: Insert Student using JPG image
        student_jpg = Student(
            name="Alice Smith",
            visual_info_path="storage/student_faces/alice.jpg",
            media_type="image"
        )
        
        # Step 4: Insert Student using MP4 video clip
        student_mp4 = Student(
            name="Bob Johnson",
            visual_info_path="storage/student_faces/bob.mp4",
            media_type="video"
        )
        
        db.add_all([student_jpg, student_mp4])
        db.commit()
        
        # Refresh to populate auto-generated primary keys (ids)
        db.refresh(student_jpg)
        db.refresh(student_mp4)

        # Step 5: Add Attendance records with optional CCTV source tracking
        att_alice = Attendance(
            student_id=student_jpg.id,
            date=date.today(),
            marked_status="Present",
            cctv_status="detected",
            cctv_source_path="storage/cctv_logs/frame_alice.jpg"
        )
        
        att_bob = Attendance(
            student_id=student_mp4.id,
            date=date.today(),
            marked_status="Absent",
            cctv_status="not detected",
            cctv_source_path="storage/cctv_logs/clip_feed.mp4"
        )

        db.add_all([att_alice, att_bob])
        db.commit()

        # Step 6: Query and print results to confirm integrity
        records = db.query(Student).all()
        print("=== DATABASE SCHEMA VERIFICATION SUCCESSFUL ===\n")
        for s in records:
            print(f"Student ID: {s.id} | Name: {s.name} | Path: {s.visual_info_path} | Type: {s.media_type}")
            for a in s.attendances:
                print(f"  └── Date: {a.date} | Status: {a.marked_status} | CCTV: {a.cctv_status} | CCTV Ref: {a.cctv_source_path}")
            print("-" * 75)

    except Exception as e:
        print(f"Verification Failed: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    verify_database_schema()