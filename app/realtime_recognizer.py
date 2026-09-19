import cv2
import json
import os
import numpy as np
import face_recognition
import datetime
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models import Student, Attendance
from app.config import STORAGE_DIR
from app.face_engine import extract_face_encoding


def recognize_webcam_frame(frame_bytes: bytes, db: Session, target_date=None):
    """Match one browser webcam frame and mark recognized students as detected."""
    if target_date is None:
        target_date = datetime.date.today()

    frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
    frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("The webcam frame could not be decoded.")

    students = db.query(Student).all()
    known_students = []
    for student in students:
        try:
            encoding_json = student.face_encoding
            if not encoding_json:
                stored_path = student.face_image_path or student.visual_info_path
                if not stored_path:
                    continue
                if not os.path.isabs(stored_path):
                    stored_path = os.path.join(STORAGE_DIR, os.path.basename(stored_path))
                encoding_json = extract_face_encoding(stored_path, student.media_type or "image")
            known_students.append((student, np.array(json.loads(encoding_json))))
        except (TypeError, json.JSONDecodeError, ValueError):
            continue
    if not known_students:
        return []

    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    known_encodings = [encoding for _, encoding in known_students]
    detected_students = {}

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
        distances = face_recognition.face_distance(known_encodings, face_encoding)
        if len(distances) == 0:
            continue
        best_index = int(np.argmin(distances))
        if matches[best_index]:
            student = known_students[best_index][0]
            detected_students[student.id] = student

    for student in detected_students.values():
        attendance = db.query(Attendance).filter(
            Attendance.student_id == student.id,
            Attendance.date == target_date,
        ).first()
        if not attendance:
            attendance = Attendance(student_id=student.id, date=target_date)
            db.add(attendance)
        attendance.cctv_status = "detected"
        attendance.marked_status = "Present"
        attendance.verification_status = "DETECTED"
        attendance.cctv_source_path = "browser-webcam"

    if detected_students:
        db.commit()

    return [{"id": student.id, "name": student.name} for student in detected_students.values()]

def run_live_opencv_recognition(video_source=0, target_date=None):
    """
    Opens an OpenCV video feed (0 for webcam or 'path/to/video.mp4'),
    queries student face encodings from DB, and updates attendance live.
    """
    if target_date is None:
        target_date = datetime.date.today().strftime("%Y-%m-%d")

    db: Session = SessionLocal()

    # Step 1: Load all student encodings from DB into memory
    students = db.query(Student).all()
    if not students:
        print("No student face data found in database.")
        db.close()
        return

    known_face_ids = []
    known_face_encodings = []
    known_face_names = []

    for student in students:
        # Convert stored JSON string back to NumPy array
        encoding_array = np.array(json.loads(student.face_encoding))
        known_face_encodings.append(encoding_array)
        known_face_ids.append(student.id)
        known_face_names.append(student.name)

    print(f"Loaded {len(known_face_encodings)} student face templates from database.")

    # Step 2: Initialize OpenCV Video Stream
    video_capture = cv2.VideoCapture(video_source)
    process_this_frame = True

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break

        # Resize frame to 1/4 size for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Process every alternate frame to maintain high FPS
        if process_this_frame:
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names_in_frame = []

            for face_encoding in face_encodings:
                # Calculate Euclidean distance against stored DB vectors
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

                name = "Unknown"
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        matched_student_id = known_face_ids[best_match_index]
                        name = known_face_names[best_match_index]

                        # Step 3: Update Database Attendance Status
                        update_attendance_db(db, matched_student_id, target_date)

                face_names_in_frame.append(name)

        process_this_frame = not process_this_frame

        # Step 4: Render OpenCV Overlay Bounding Boxes & Names
        for (top, right, bottom, left), name in zip(face_locations, face_names_in_frame):
            # Scale back up by 4x to match original frame size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

            # Draw box around face
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            # Draw label background
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            # Write student name
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        # Display output feed
        cv2.imshow("VisionAttend - Live OpenCV Recognition", frame)

        # Press 'q' to exit live recognition loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    db.close()


def update_attendance_db(db: Session, student_id: int, date_str: str):
    """
    Checks if an attendance record exists for the given student and date,
    then updates status from PENDING to DETECTED in SQLite.
    """
    record = db.query(Attendance).filter(
        Attendance.student_id == student_id,
        Attendance.date == date_str
    ).first()

    if record and record.verification_status != "DETECTED":
        record.verification_status = "DETECTED"
        record.verification_time = datetime.datetime.utcnow()
        db.commit()
        print(f"[DB SUCCESS] Student ID {student_id} verified as DETECTED in database.")