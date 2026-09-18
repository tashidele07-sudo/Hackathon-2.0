import os
import shutil
import datetime
from fastapi import FastAPI, Depends, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.database import engine, Base, get_db
from app.models import Student, Attendance
from app.config import STORAGE_DIR
from app.face_engine import extract_face_encoding
from app.cctv_verifier import process_cctv_verification

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="VisionAttend - CCTV Facial Verification System")

# Serve static files & templates
app.mount("/static", StaticFiles(directory=STORAGE_DIR), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
def render_dashboard(request: Request, db: Session = Depends(get_db)):
    students = db.query(Student).all()
    attendances = db.query(Attendance).order_by(Attendance.entry_time.desc()).all()
    
    total_students = len(students)
    today = datetime.date.today().strftime("%Y-%m-%d")
    today_attendances = [a for a in attendances if a.date == today]
    detected_count = sum(1 for a in today_attendances if a.verification_status == "DETECTED")
    not_detected_count = sum(1 for a in today_attendances if a.verification_status == "NOT_DETECTED")
    pending_count = sum(1 for a in today_attendances if a.verification_status == "PENDING")

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "students": students,
        "attendances": attendances,
        "metrics": {
            "total_students": total_students,
            "today": today,
            "detected": detected_count,
            "not_detected": not_detected_count,
            "pending": pending_count
        }
    })


@app.post("/api/students")
async def register_student(
    name: str = Form(...),
    roll_number: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    existing = db.query(Student).filter(Student.roll_number == roll_number).first()
    if existing:
        raise HTTPException(status_code=400, detail="Student with this Roll Number already exists.")

    # Save visual face file to disk
    file_extension = file.filename.split(".")[-1]
    file_name = f"{roll_number}.{file_extension}"
    file_path = os.path.join(STORAGE_DIR, file_name)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Extract face features
    try:
        encoding_json = extract_face_encoding(file_path)
    except Exception as e:
        os.remove(file_path)
        raise HTTPException(status_code=400, detail=str(e))

    student = Student(
        name=name,
        roll_number=roll_number,
        face_image_path=file_name,
        face_encoding=encoding_json
    )
    db.add(student)
    db.commit()
    db.refresh(student)
    return {"message": "Student registered successfully", "id": student.id}


@app.post("/api/attendance/entry")
def mark_attendance(
    student_id: int = Form(...),
    date_str: str = Form(...),
    db: Session = Depends(get_db)
):
    existing = db.query(Attendance).filter(
        Attendance.student_id == student_id, Attendance.date == date_str
    ).first()
    
    if existing:
        raise HTTPException(status_code=400, detail="Attendance already logged for this date.")

    attendance = Attendance(
        student_id=student_id,
        date=date_str,
        verification_status="PENDING"
    )
    db.add(attendance)
    db.commit()
    return {"message": "Attendance entry recorded. Status set to PENDING verification."}


@app.post("/api/attendance/verify-cctv")
async def run_cctv_verification(
    date_str: str = Form(...),
    cctv_file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    # Save temporary CCTV video file
    temp_video_path = os.path.join(STORAGE_DIR, f"temp_cctv_{cctv_file.filename}")
    with open(temp_video_path, "wb") as buffer:
        shutil.copyfileobj(cctv_file.file, buffer)

    try:
        result = process_cctv_verification(temp_video_path, date_str, db)
    finally:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

    return result