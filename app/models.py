import datetime
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from app.database import Base

class Student(Base):
    __tablename__ = "students"

    id = Column(Integer, primary_key=True, index=True)
    roll_number = Column(String(50), unique=True, index=True, nullable=False)
    name = Column(String(100), nullable=False)
    face_image_path = Column(String(255), nullable=False)
    face_encoding = Column(Text, nullable=False)  # JSON-encoded 128D face vector
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    attendances = relationship("Attendance", back_populates="student")

class Attendance(Base):
    __tablename__ = "attendance"

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey("students.id"), nullable=False)
    date = Column(String(10), index=True, nullable=False)  # YYYY-MM-DD
    entry_time = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Verification status: 'PENDING', 'DETECTED', 'NOT_DETECTED'
    verification_status = Column(String(20), default="PENDING")
    verification_time = Column(DateTime, nullable=True)

    student = relationship("Student", back_populates="attendances")