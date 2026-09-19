# app/cctv_verifier.py
import cv2
from datetime import date
from sqlalchemy.orm import Session
from app.models import Student, Attendance
from app.face_engine import extract_face_encoding, match_face_in_frame

def process_cctv_video(cctv_video_path: str, db: Session, frame_skip: int = 15):
    """
    Reads a CCTV video (.mp4), checks frames against database students,
    and updates Attendance table with cctv_status='detected'.
    """
    cap = cv2.VideoCapture(cctv_video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open CCTV feed: {cctv_video_path}")

    today = date.today()
    students = db.query(Student).all()
    
    # Pre-fetch and cache student face encodings
    student_cache = []
    for s in students:
        try:
            if s.face_encoding:
                encoding = s.face_encoding
            else:
                encoding = extract_face_encoding(s.visual_info_path, s.media_type)
            # Ensure attendance row exists for today
            att = db.query(Attendance).filter(
                Attendance.student_id == s.id, 
                Attendance.date == today
            ).first()
            if not att:
                att = Attendance(student_id=s.id, date=today)
                db.add(att)
                db.commit()
                db.refresh(att)
            
            student_cache.append({"student_id": s.id, "encoding": encoding, "attendance": att})
        except Exception as e:
            print(f"Skipping student {s.name}: {e}")

    frame_count = 0
    detected_ids = set()

    # Loop through CCTV video stream
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        # Process every Nth frame to optimize performance
        if frame_count % frame_skip != 0:
            continue

        for item in student_cache:
            s_id = item["student_id"]
            if s_id in detected_ids:
                continue  # Already detected, skip checking again

            if match_face_in_frame(frame, item["encoding"]):
                detected_ids.add(s_id)
                att = item["attendance"]
                att.cctv_status = "detected"
                att.marked_status = "Present"
                att.verification_status = "DETECTED"
                att.cctv_source_path = cctv_video_path
                db.commit()

    cap.release()
    return {"processed_frames": frame_count, "detected_student_ids": list(detected_ids)}