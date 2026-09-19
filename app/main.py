# app/main.py
import os
import shutil
import datetime
from fastapi import FastAPI, Depends, HTTPException, Form, UploadFile, File, Request
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.database import engine, Base, SessionLocal, migrate_legacy_schema
from app.models import Student, Attendance
from app.config import STORAGE_DIR
from app.face_engine import extract_face_encoding
from app.cctv_verifier import process_cctv_video
from app.realtime_recognizer import recognize_webcam_frame

migrate_legacy_schema()

app = FastAPI(title="Vision Attendance API")
app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("VISION_ATTEND_SECRET", "change-this-development-secret"),
)

# Mount storage directory so dashboard <img src="/static/101.jpg"> loads face thumbnails
app.mount("/static", StaticFiles(directory=STORAGE_DIR), name="static")
templates = Jinja2Templates(directory="templates")

LCID_PREFIX = "LC0001700"

def ensure_today_attendance(db: Session, students):
    today = datetime.date.today()
    student_ids = {student.id for student in students}
    existing_ids = {
        student_id
        for (student_id,) in db.query(Attendance.student_id)
        .filter(
            Attendance.date == today,
            Attendance.student_id.in_(student_ids),
        )
        .all()
    }

    missing_attendance = [
        Attendance(
            student_id=student.id,
            date=today,
            marked_status="Absent",
            cctv_status="not detected",
            verification_status="PENDING",
        )
        for student in students
        if student.id not in existing_ids
    ]
    if missing_attendance:
        db.add_all(missing_attendance)
        db.commit()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def require_admin(request: Request):
    if not request.session.get("admin_authenticated"):
        if not request.url.path.startswith("/api/"):
            raise HTTPException(status_code=307, headers={"Location": "/login"})
        raise HTTPException(status_code=401, detail="Admin login required.")

@app.get("/")
def root():
    return RedirectResponse(url="/dashboard")


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    if request.session.get("admin_authenticated"):
        return RedirectResponse(url="/dashboard")
    return templates.TemplateResponse(
        request=request,
        name="login.html",
        context={"error": None},
    )


@app.post("/login", response_class=HTMLResponse)
def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
):
    expected_username = os.getenv("VISION_ATTEND_ADMIN_USER", "admin")
    expected_password = os.getenv("VISION_ATTEND_ADMIN_PASSWORD", "admin")
    if username != expected_username or password != expected_password:
        return templates.TemplateResponse(
            request=request,
            name="login.html",
            context={"error": "Invalid username or password."},
            status_code=401,
        )
    request.session["admin_authenticated"] = True
    return RedirectResponse(url="/dashboard", status_code=303)


@app.post("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/login", status_code=303)

@app.get("/dashboard", response_class=HTMLResponse)
def render_dashboard(
    request: Request,
    db: Session = Depends(get_db),
    _: None = Depends(require_admin),
):
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    
    students = db.query(Student).order_by(Student.name).all()
    ensure_today_attendance(db, students)
    attendances = (
        db.query(Attendance)
        .filter(Attendance.date == datetime.date.today())
        .join(Student)
        .order_by(Student.name)
        .all()
    )
    student_attendance_status = {
        attendance.student_id: attendance.verification_status
        for attendance in attendances
    }

    metrics = {
        "today": today_str,
        "total_students": len(students),
        "detected": sum(1 for a in attendances if a.verification_status == "DETECTED"),
        "not_detected": sum(1 for a in attendances if a.verification_status == "NOT_DETECTED"),
        "pending": sum(1 for a in attendances if a.verification_status == "PENDING"),
    }
    requested_student_step = request.query_params.get("student_step", "register")
    if requested_student_step not in {"register", "attendance", "cctv", "directory"}:
        requested_student_step = "register"
    active_tab = request.query_params.get("active_tab", "dashboard")
    if active_tab not in {"dashboard", "students", "attendance", "cctv", "staff", "support"}:
        active_tab = "dashboard"

    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "metrics": metrics,
            "next_lcid": LCID_PREFIX,
            "students": students,
            "attendances": attendances,
            "student_attendance_status": student_attendance_status,
            "student_step": requested_student_step,
            "active_tab": active_tab,
            "notification": request.session.pop("notification", None),
        },
    )

@app.post("/api/students")
def register_student(
    request: Request,
    name: str = Form(...),
    lcid_suffix: str = Form(...),
    file: UploadFile = File(...),
    profile_file: UploadFile = File(...),
    db: Session = Depends(get_db),
    _: None = Depends(require_admin),
):
    if len(lcid_suffix) != 4 or not lcid_suffix.isdigit():
        raise HTTPException(
            status_code=400,
            detail="LCID suffix must contain exactly 4 digits.",
        )

    lcid = f"{LCID_PREFIX}{lcid_suffix}"
    if db.query(Student).filter(Student.lcid == lcid).first():
        request.session["notification"] = {
            "type": "warning",
            "title": "LCID already registered",
            "message": f"LCID {lcid} is already assigned to an enrolled student. Please enter a different LCID.",
        }
        return RedirectResponse(url="/dashboard?student_step=register", status_code=303)
    filename = file.filename or "upload"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "bin"
    media_type = "video" if ext in {"mp4", "avi", "mov", "mkv", "webm"} else "image"
    stored_filename = f"{lcid}.{ext}"
    destination_path = os.path.join(STORAGE_DIR, stored_filename)
    profile_filename = profile_file.filename or "profile.jpg"
    profile_ext = profile_filename.rsplit(".", 1)[-1].lower() if "." in profile_filename else "jpg"
    if profile_ext not in {"jpg", "jpeg", "png", "webp"}:
        raise HTTPException(status_code=400, detail="Student profile must be a JPG, PNG, or WEBP image.")
    stored_profile_filename = f"{lcid}_profile.{profile_ext}"
    profile_path = os.path.join(STORAGE_DIR, stored_profile_filename)

    with open(destination_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        encoding_json = extract_face_encoding(destination_path, media_type)
    except Exception as e:
        if os.path.exists(destination_path):
            os.remove(destination_path)
        raise HTTPException(status_code=400, detail=f"Face extraction failed: {e}")

    with open(profile_path, "wb") as buffer:
        shutil.copyfileobj(profile_file.file, buffer)

    student = Student(
        name=name,
        lcid=lcid,
        profile_image_path=stored_profile_filename,
        face_image_path=stored_filename,
        face_encoding=encoding_json,
        visual_info_path=stored_filename,
        media_type=media_type,
    )
    db.add(student)
    db.commit()
    db.refresh(student)

    db.add(
        Attendance(
            student_id=student.id,
            date=datetime.date.today(),
            marked_status="Absent",
            cctv_status="not detected",
            verification_status="PENDING",
        )
    )
    db.commit()
    request.session["notification"] = {
        "type": "success",
        "title": "Student added",
        "message": f"{name} was successfully added to the enrolled directory.",
    }

    return RedirectResponse(url="/dashboard?student_step=attendance", status_code=303)

@app.post("/api/attendance/entry")
def manual_attendance_entry(
    request: Request,
    student_id: int = Form(...),
    date_str: str = Form(...),
    db: Session = Depends(get_db),
    _: None = Depends(require_admin),
):
    try:
        attendance_date = datetime.date.fromisoformat(date_str)
    except ValueError:
        raise HTTPException(status_code=400, detail="Please provide a valid attendance date.")

    if not db.query(Student).filter(Student.id == student_id).first():
        raise HTTPException(status_code=404, detail="Selected student was not found.")

    existing = db.query(Attendance).filter(
        Attendance.student_id == student_id,
        Attendance.date == attendance_date
    ).first()

    if not existing:
        record = Attendance(
            student_id=student_id,
            date=attendance_date,
            marked_status="Absent",
            cctv_status="not detected",
            verification_status="PENDING",
        )
        try:
            db.add(record)
            db.commit()
        except SQLAlchemyError:
            db.rollback()
            raise HTTPException(
                status_code=500,
                detail="Could not save the attendance entry. Please try again.",
            )

    return RedirectResponse(url="/dashboard?student_step=cctv", status_code=303)

@app.post("/api/attendance/verify-cctv")
def verify_cctv_form(
    request: Request,
    date_str: str = Form(...),
    cctv_file: UploadFile = File(...),
    db: Session = Depends(get_db),
    _: None = Depends(require_admin),
):
    temp_video_path = os.path.join(STORAGE_DIR, f"temp_{cctv_file.filename}")
    with open(temp_video_path, "wb") as buffer:
        shutil.copyfileobj(cctv_file.file, buffer)

    try:
        verification_date = datetime.date.fromisoformat(date_str)
    except ValueError:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        raise HTTPException(status_code=400, detail="Please provide a valid CCTV date.")

    result = process_cctv_video(
        cctv_video_path=temp_video_path,
        db=db,
        target_date=verification_date,
    )

    if os.path.exists(temp_video_path):
        os.remove(temp_video_path)

    detected_count = len(result["detected_student_ids"])
    not_detected_students = (
        db.query(Student)
        .filter(Student.id.in_(result["not_detected_student_ids"]))
        .order_by(Student.name)
        .all()
    )
    if not_detected_students:
        names = ", ".join(student.name.strip() for student in not_detected_students)
        request.session["notification"] = {
            "type": "warning",
            "title": "Face not detected",
            "message": f"{len(not_detected_students)} student(s) were not found in the CCTV video: {names}.",
        }
    else:
        request.session["notification"] = {
            "type": "success",
            "title": "CCTV verification complete",
            "message": f"Faces detected for all registered students ({detected_count}).",
        }

    return RedirectResponse(url="/dashboard?active_tab=attendance", status_code=303)


@app.post("/api/attendance/verify-webcam-frame")
def verify_webcam_frame(
    date_str: str = Form(...),
    webcam_frame: UploadFile = File(...),
    db: Session = Depends(get_db),
    _: None = Depends(require_admin),
):
    try:
        target_date = datetime.date.fromisoformat(date_str)
    except ValueError:
        raise HTTPException(status_code=400, detail="Please provide a valid webcam date.")

    try:
        detected_students = recognize_webcam_frame(webcam_frame.file.read(), db, target_date)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

    return JSONResponse({"detected_students": detected_students})


@app.post("/api/attendance/complete-webcam-verify")
def complete_webcam_verification(
    request: Request,
    date_str: str = Form(...),
    detected_student_ids: list[int] = Form(default=[]),
    db: Session = Depends(get_db),
    _: None = Depends(require_admin),
):
    try:
        target_date = datetime.date.fromisoformat(date_str)
    except ValueError:
        raise HTTPException(status_code=400, detail="Please provide a valid webcam date.")

    students = db.query(Student).all()
    detected_ids = set(detected_student_ids)
    proxy_count = 0
    for student in students:
        attendance = db.query(Attendance).filter(
            Attendance.student_id == student.id,
            Attendance.date == target_date,
        ).first()
        if not attendance:
            attendance = Attendance(student_id=student.id, date=target_date)
            db.add(attendance)
        if student.id not in detected_ids:
            attendance.cctv_status = "not detected"
            attendance.marked_status = "Absent"
            attendance.verification_status = "NOT_DETECTED"
            attendance.cctv_source_path = "browser-webcam"
            proxy_count += 1

    db.commit()
    request.session["notification"] = {
        "type": "warning" if proxy_count else "success",
        "title": "Webcam verification complete",
        "message": f"{proxy_count} student(s) marked as Proxy." if proxy_count else "All enrolled students were verified.",
    }
    return JSONResponse({"proxy_count": proxy_count, "redirect_url": "/dashboard?active_tab=attendance"})


@app.post("/api/attendance/{attendance_id}/delete")
def delete_attendance(
    attendance_id: int,
    request: Request,
    return_page: str = Form("dashboard"),
    db: Session = Depends(get_db),
    _: None = Depends(require_admin),
):
    attendance = db.query(Attendance).filter(Attendance.id == attendance_id).first()
    if not attendance:
        raise HTTPException(status_code=404, detail="Attendance entry was not found.")
    db.delete(attendance)
    db.commit()
    request.session["notification"] = {
        "type": "success",
        "title": "Attendance record deleted",
        "message": "The verification record was permanently removed.",
    }
    redirect_url = "/dashboard?active_tab=attendance" if return_page == "attendance" else "/dashboard"
    return RedirectResponse(url=redirect_url, status_code=303)


@app.post("/api/students/{student_id}/delete")
def delete_student(
    student_id: int,
    request: Request,
    return_page: str = Form("dashboard"),
    db: Session = Depends(get_db),
    _: None = Depends(require_admin),
):
    student = db.query(Student).filter(Student.id == student_id).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student was not found.")

    for filename in {
        student.profile_image_path,
        student.face_image_path,
        student.visual_info_path,
    }:
        if filename:
            file_path = os.path.join(STORAGE_DIR, filename)
            if os.path.exists(file_path):
                os.remove(file_path)

    db.delete(student)
    db.commit()
    request.session["notification"] = {
        "type": "success",
        "title": "Student deleted",
        "message": "The student profile, face data, photo, and attendance records were permanently removed.",
    }
    redirect_url = "/dashboard?active_tab=students&student_step=directory" if return_page == "students_directory" else "/dashboard"
    return RedirectResponse(url=redirect_url, status_code=303)