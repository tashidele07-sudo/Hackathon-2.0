# app/schemas.py
from pydantic import BaseModel
from datetime import date
from typing import Optional, List

class StudentBase(BaseModel):
    name: str
    visual_info_path: str
    media_type: str = "image"

class StudentResponse(StudentBase):
    id: int

    class Config:
        from_attributes = True

class AttendanceResponse(BaseModel):
    id: int
    student_id: int
    student_name: str
    visual_info_path: str
    date: date
    marked_status: str
    cctv_status: str
    cctv_source_path: Optional[str] = None

    class Config:
        from_attributes = True

class DashboardSummaryResponse(BaseModel):
    total_students: int
    marked_present: int
    cctv_detected: int
    date: date

class CCTVVerifyRequest(BaseModel):
    cctv_video_path: str