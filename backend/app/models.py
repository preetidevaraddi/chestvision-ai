import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    Column, String, Integer, Float, DateTime, ForeignKey, Text, Enum, Boolean
)
from sqlalchemy.orm import relationship

from app.database import Base


def gen_uuid() -> str:
    return str(uuid.uuid4())


class GenderEnum(str, enum.Enum):
    male = "male"
    female = "female"
    other = "other"


class ReportStatus(str, enum.Enum):
    pending = "pending"
    viewed = "viewed"
    review = "review"


class UncertaintyStatus(str, enum.Enum):
    high_confidence = "High Confidence"
    moderate_confidence = "Moderate Confidence"
    requires_review = "Requires Review"


class PriorityLevel(str, enum.Enum):
    normal = "Normal"
    high = "High"
    urgent = "Urgent"


class Admin(Base):
    __tablename__ = "admins"

    admin_id = Column(String(36), primary_key=True, default=gen_uuid)
    name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    patients = relationship("Patient", back_populates="registered_by")


class Doctor(Base):
    __tablename__ = "doctors"

    doctor_id = Column(String(36), primary_key=True, default=gen_uuid)
    name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class Patient(Base):
    __tablename__ = "patients"

    patient_id = Column(String(36), primary_key=True, default=gen_uuid)
    name = Column(String(255), nullable=False)
    age = Column(Integer, nullable=False)
    gender = Column(Enum(GenderEnum), nullable=False)
    contact_number = Column(String(50), nullable=True)
    address = Column(String(500), nullable=True)
    registered_by_id = Column(String(36), ForeignKey("admins.admin_id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    registered_by = relationship("Admin", back_populates="patients")
    xray_images = relationship("XrayImage", back_populates="patient", cascade="all, delete-orphan")


class XrayImage(Base):
    __tablename__ = "xray_images"

    image_id = Column(String(36), primary_key=True, default=gen_uuid)
    patient_id = Column(String(36), ForeignKey("patients.patient_id"), nullable=False)
    image_path = Column(String(1000), nullable=False)
    original_filename = Column(String(500), nullable=False)
    uploaded_by_id = Column(String(36), ForeignKey("admins.admin_id"))
    uploaded_at = Column(DateTime, default=datetime.utcnow)

    patient = relationship("Patient", back_populates="xray_images")
    analysis_results = relationship("AnalysisResult", back_populates="xray_image", cascade="all, delete-orphan")


class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    result_id = Column(String(36), primary_key=True, default=gen_uuid)
    image_id = Column(String(36), ForeignKey("xray_images.image_id"), nullable=False)

    # Predicted conditions stored as JSON text: [{"label": "...", "confidence": 0.92}, ...]
    predicted_conditions = Column(Text, nullable=False)
    top_prediction_label = Column(String(255), nullable=False)
    top_prediction_confidence = Column(Float, nullable=False)

    uncertainty_status = Column(Enum(UncertaintyStatus), nullable=False)
    priority = Column(Enum(PriorityLevel), nullable=False)

    heatmap_path = Column(String(1000), nullable=True)  # suspicious-region visualization
    model_version = Column(String(100), nullable=True)  # "trained" vs "demo_fallback" - kept distinct
    is_demo_model = Column(Boolean, nullable=False, default=True)  # True = NOT clinically meaningful

    analyzed_at = Column(DateTime, default=datetime.utcnow)

    xray_image = relationship("XrayImage", back_populates="analysis_results")
    report = relationship("Report", back_populates="analysis_result", uselist=False, cascade="all, delete-orphan")


class Report(Base):
    __tablename__ = "reports"

    report_id = Column(String(36), primary_key=True, default=gen_uuid)
    analysis_result_id = Column(String(36), ForeignKey("analysis_results.result_id"), nullable=False)
    report_path = Column(String(1000), nullable=False)  # generated PDF path
    generated_by_id = Column(String(36), ForeignKey("admins.admin_id"))
    generated_at = Column(DateTime, default=datetime.utcnow)
    status = Column(Enum(ReportStatus), nullable=False, default=ReportStatus.pending)

    analysis_result = relationship("AnalysisResult", back_populates="report")
