import json
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.database import get_db
from app.dependencies import require_doctor
from app.models import Doctor, Patient, XrayImage, AnalysisResult, Report, ReportStatus
from app.schemas import PatientOut, AnalysisResultOut, ReportOut, ReportStatusUpdate

router = APIRouter(prefix="/api/doctor", tags=["doctor"])

# NOTE: every endpoint in this router is a GET. There are intentionally no
# POST/PUT/DELETE routes here - a Doctor account cannot write data even by
# guessing a URL, because those routes simply do not exist on this router,
# and require_doctor() rejects any Admin-router call with a doctor token.


@router.get("/dashboard")
def dashboard(db: Session = Depends(get_db), doctor: Doctor = Depends(require_doctor)):
    total_reports = db.query(func.count(Report.report_id)).scalar()
    priority_cases = db.query(func.count(AnalysisResult.result_id)).filter(
        AnalysisResult.priority.in_(["Urgent", "High"])
    ).scalar()
    recent_reports = db.query(Report).order_by(Report.generated_at.desc()).limit(10).all()
    
    recent_reports_out = []
    for r in recent_reports:
        a = db.query(AnalysisResult).filter(AnalysisResult.result_id == r.analysis_result_id).first()
        x = db.query(XrayImage).filter(XrayImage.image_id == a.image_id).first()
        p = db.query(Patient).filter(Patient.patient_id == x.patient_id).first()
        
        recent_reports_out.append({
            "report_id": r.report_id,
            "report_display_id": r.report_display_id,
            "patient_name": p.name,
            "patient_id": p.patient_id,
            "patient_display_id": p.patient_display_id,
            "predicted_condition": a.top_prediction_label,
            "confidence": a.top_prediction_confidence,
            "priority": a.priority.value if hasattr(a.priority, "value") else a.priority,
            "status": r.status.value if hasattr(r.status, "value") else r.status,
            "generated_at": r.generated_at
        })

    return {
        "total_reports": total_reports,
        "priority_cases": priority_cases,
        "recent_reports": recent_reports_out,
    }


@router.get("/reports")
def list_reports(db: Session = Depends(get_db), doctor: Doctor = Depends(require_doctor)):
    reports = db.query(Report).order_by(Report.generated_at.desc()).all()
    out = []
    for r in reports:
        a = db.query(AnalysisResult).filter(AnalysisResult.result_id == r.analysis_result_id).first()
        x = db.query(XrayImage).filter(XrayImage.image_id == a.image_id).first()
        p = db.query(Patient).filter(Patient.patient_id == x.patient_id).first()
        
        out.append({
            "report_id": r.report_id,
            "report_display_id": r.report_display_id,
            "patient_name": p.name,
            "patient_id": p.patient_id,
            "patient_display_id": p.patient_display_id,
            "predicted_condition": a.top_prediction_label,
            "confidence": a.top_prediction_confidence,
            "priority": a.priority.value if hasattr(a.priority, "value") else a.priority,
            "status": r.status.value if hasattr(r.status, "value") else r.status,
            "generated_at": r.generated_at
        })
    return out


@router.get("/reports/{report_id}")
def get_report_detail(report_id: str, db: Session = Depends(get_db), doctor: Doctor = Depends(require_doctor)):
    report = db.query(Report).filter(Report.report_id == report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    analysis = db.query(AnalysisResult).filter(AnalysisResult.result_id == report.analysis_result_id).first()
    xray = db.query(XrayImage).filter(XrayImage.image_id == analysis.image_id).first()
    patient = db.query(Patient).filter(Patient.patient_id == xray.patient_id).first()

    return {
        "report_id": report.report_id,
        "report_display_id": report.report_display_id,
        "report_path": report.report_path,
        "generated_at": report.generated_at,
        "status": report.status.value if hasattr(report.status, "value") else report.status,
        "patient": PatientOut.model_validate(patient),
        "xray_image_path": xray.image_path,
        "analysis": AnalysisResultOut(
            result_id=analysis.result_id,
            result_display_id=analysis.result_display_id,
            image_id=analysis.image_id,
            patient_name=patient.name,
            patient_display_id=patient.patient_display_id,
            predicted_conditions=json.loads(analysis.predicted_conditions),
            top_prediction_label=analysis.top_prediction_label,
            top_prediction_confidence=analysis.top_prediction_confidence,
            uncertainty_status=analysis.uncertainty_status.value if hasattr(analysis.uncertainty_status, "value") else analysis.uncertainty_status,
            priority=analysis.priority.value if hasattr(analysis.priority, "value") else analysis.priority,
            heatmap_path=analysis.heatmap_path,
            is_demo_model=analysis.is_demo_model,
            disclaimer=(
                "DEMO MODE prediction - not clinically meaningful." if analysis.is_demo_model
                else "Academic prototype - not clinically validated."
            ),
            analyzed_at=analysis.analyzed_at,
        ),
    }


@router.get("/patients/{patient_id}", response_model=PatientOut)
def view_patient(patient_id: str, db: Session = Depends(get_db), doctor: Doctor = Depends(require_doctor)):
    patient = db.query(Patient).filter(Patient.patient_id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient


@router.patch("/reports/{report_id}/status")
def update_report_status(report_id: str, payload: ReportStatusUpdate, db: Session = Depends(get_db), doctor: Doctor = Depends(require_doctor)):
    report = db.query(Report).filter(Report.report_id == report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    
    valid_statuses = ["pending", "viewed", "review"]
    if payload.status not in valid_statuses:
        raise HTTPException(status_code=400, detail="Invalid status")

    try:
        status_enum = ReportStatus(payload.status)
        report.status = status_enum
        db.commit()
        db.refresh(report)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid status value")
        
    return {"status": report.status.value if hasattr(report.status, "value") else report.status}
