from datetime import datetime
from typing import List, Optional
import re
from pydantic import BaseModel, EmailStr, field_validator


# ---------- Auth ----------

class AdminRegister(BaseModel):
    name: str
    email: EmailStr
    password: str
    confirm_password: str

    @field_validator("password")
    @classmethod
    def validate_password_strength(cls, v):
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        if not re.search(r"[A-Z]", v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not re.search(r"[a-z]", v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not re.search(r"\d", v):
            raise ValueError("Password must contain at least one number")
        if not re.search(r"[!@#$%^&*(),.?\":{}|<>_\-+=\[\]\\;'/`~]", v):
            raise ValueError("Password must contain at least one special character (e.g. ! @ # $ %)")
        return v

    @field_validator("confirm_password")
    @classmethod
    def passwords_match(cls, v, info):
        if "password" in info.data and v != info.data["password"]:
            raise ValueError("Passwords do not match")
        return v


class DoctorRegister(AdminRegister):
    pass


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    name: str


# ---------- Patient ----------

class PatientCreate(BaseModel):
    name: str
    age: int
    gender: str
    contact_number: Optional[str] = None
    address: Optional[str] = None


class PatientUpdate(BaseModel):
    name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    contact_number: Optional[str] = None
    address: Optional[str] = None


class PatientOut(BaseModel):
    patient_id: str
    name: str
    age: int
    gender: str
    contact_number: Optional[str]
    address: Optional[str]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# ---------- X-ray / Analysis ----------

class XrayImageOut(BaseModel):
    image_id: str
    patient_id: str
    image_path: str
    uploaded_at: datetime

    class Config:
        from_attributes = True


class ConditionScore(BaseModel):
    label: str
    confidence: float


class AnalysisResultOut(BaseModel):
    result_id: str
    image_id: str
    predicted_conditions: List[ConditionScore]
    top_prediction_label: str
    top_prediction_confidence: float
    uncertainty_status: str
    priority: str
    heatmap_path: Optional[str]
    is_demo_model: bool
    disclaimer: str
    analyzed_at: datetime

    class Config:
        from_attributes = True


class ReportOut(BaseModel):
    report_id: str
    analysis_result_id: str
    report_path: str
    generated_at: datetime
    status: str = "pending"

    class Config:
        from_attributes = True


class ReportStatusUpdate(BaseModel):
    status: str
