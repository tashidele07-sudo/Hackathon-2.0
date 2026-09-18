from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class StudentCreate(BaseModel):
    roll_number: str
    name: str

class AttendanceEntry(BaseModel):
    student_id: int
    date: str  # YYYY-MM-DD

class AttendanceResponse(BaseModel):
    id: int
    student_id: int
    student_name: str
    roll_number: str
    date: str
    entry_time: datetime
    verification_status: str
    verification_time: Optional[datetime]

    class Config:
        from_attributes = True