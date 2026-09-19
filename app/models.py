from datetime import date
from sqlalchemy import Column, Integer, String, Date, ForeignKey
from sqlalchemy.orm import relationship
from app.database import Base  # Updated import path


class Student(Base):
    __tablename__ = "students"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    lcid = Column(String(50), nullable=True, unique=True, index=True)
    profile_image_path = Column(String(255), nullable=True)
    face_image_path = Column(String(255), nullable=True)
    face_encoding = Column(String, nullable=True)
    # Legacy fields retained for existing databases and older scripts.
    visual_info_path = Column(String(255), nullable=True)
    media_type = Column(String(10), nullable=True, default="image")

    attendances = relationship(
        "Attendance", back_populates="student", cascade="all, delete-orphan"
    )


class Attendance(Base):
    __tablename__ = "attendances"

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey("students.id"), nullable=False)
    date = Column(Date, default=date.today, nullable=False, index=True)
    marked_status = Column(String(20), nullable=True, default="Absent")
    cctv_status = Column(String(20), nullable=True, default="not detected")
    verification_status = Column(String(20), nullable=True, default="PENDING")
    # Optional field to store the specific CCTV video (.mp4) or frame (.jpg) that triggered detection
    cctv_source_path = Column(String(255), nullable=True)

    student = relationship("Student", back_populates="attendances")