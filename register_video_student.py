import os
import cv2
import json
import datetime
import face_recognition
from app.database import SessionLocal, engine, Base
from app.models import Student, Attendance
from app.config import STORAGE_DIR

Base.metadata.create_all(bind=engine)

def register_student_from_video(video_path: str, name: str, roll_number: str):
    if not os.path.exists(video_path):
        print(f"[ERROR] Video file not found at: {video_path}")
        return

    db = SessionLocal()

    # Check if student already exists
    existing = db.query(Student).filter(Student.roll_number == roll_number).first()
    if existing:
        print(f"[EXISTS] Student Roll {roll_number} ({name}) is already in the database.")
        db.close()
        return

    print(f"Processing video '{video_path}' to extract face...")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Unable to open video file: {video_path}")
        db.close()
        return

    extracted_encoding = None
    saved_image_filename = f"{roll_number}.jpg"
    destination_img_path = os.path.join(STORAGE_DIR, saved_image_filename)

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        # Process every 5th frame for speed
        if frame_count % 5 != 0:
            continue

        # Convert OpenCV BGR frame to RGB for face_recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces in frame
        face_locations = face_recognition.face_locations(rgb_frame)
        if face_locations:
            encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            if encodings:
                extracted_encoding = encodings[0]
                # Save frame as reference photo
                cv2.imwrite(destination_img_path, frame)
                print(f"[FOUND] Detected face on frame {frame_count}. Saved reference image to {saved_image_filename}.")
                break

    cap.release()

    if extracted_encoding is None:
        print(f"[ERROR] Could not detect any faces in '{video_path}'. Ensure the face is clearly visible.")
        db.close()
        return

    # Convert 128D numpy array to JSON string for SQLite storage
    encoding_json = json.dumps(extracted_encoding.tolist())

    # 1. Save Student to DB
    student = Student(
        name=name,
        roll_number=roll_number,
        face_image_path=saved_image_filename,
        face_encoding=encoding_json
    )
    db.add(student)
    db.commit()
    db.refresh(student)

    # 2. Add PENDING attendance for today
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    attendance = Attendance(
        student_id=student.id,
        date=today_str,
        verification_status="PENDING"
    )
    db.add(attendance)
    db.commit()

    print(f"[SUCCESS] Registered {name} (Roll: {roll_number}) in Database!")
    db.close()

if __name__ == "__main__":
    # Path to video file in storage/student_faces/
    video_file_path = os.path.join(STORAGE_DIR, "Student1.mp4")
    
    # Register student using the video
    register_student_from_video(
        video_path=video_file_path,
        name="Bashista Rana",
        roll_number="13"
    )