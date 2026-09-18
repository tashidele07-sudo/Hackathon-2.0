import cv2
import datetime
from sqlalchemy.orm import Session
from app.models import Attendance, Student
from app.face_engine import match_face_in_frame

def process_cctv_verification(video_source: str, date_str: str, db: Session):
    """
    Reads a CCTV video feed or file, extracts pending attendance entries for the given date,
    and verifies whether each student's face appears in the footage.
    """
    # Fetch all pending attendance records for the given date
    pending_records = (
        db.query(Attendance)
        .filter(Attendance.date == date_str, Attendance.verification_status == "PENDING")
        .all()
    )

    if not pending_records:
        return {"message": "No pending attendance verification needed for this date."}

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to access CCTV video source: {video_source}")

    unverified_students = {rec.student_id: rec for rec in pending_records}
    frame_count = 0

    while cap.isOpened() and unverified_students:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        # Process every 10th frame to optimize performance
        if frame_count % 10 != 0:
            continue

        verified_ids = []
        for student_id, attendance_record in unverified_students.items():
            student = db.query(Student).filter(Student.id == student_id).first()
            if not student:
                continue

            # Verify face presence in current frame
            if match_face_in_frame(frame, student.face_encoding):
                attendance_record.verification_status = "DETECTED"
                attendance_record.verification_time = datetime.datetime.utcnow()
                verified_ids.append(student_id)

        # Remove verified students from active frame search
        for s_id in verified_ids:
            del unverified_students[s_id]

    cap.release()

    # Mark remaining unverified students as NOT_DETECTED
    for student_id, attendance_record in unverified_students.items():
        attendance_record.verification_status = "NOT_DETECTED"
        attendance_record.verification_time = datetime.datetime.utcnow()

    db.commit()
    return {"status": "Complete", "processed_records": len(pending_records)}