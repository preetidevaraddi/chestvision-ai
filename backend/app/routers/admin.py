import json
import os
import shutil
import uuid

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.database import get_db
from app.dependencies import require_admin
from app.core.config import settings
from app.models import Admin, Patient, XrayImage, AnalysisResult, Report, GenderEnum
from app.schemas import PatientCreate, PatientUpdate, PatientOut, AnalysisResultOut, ReportOut, XrayImageOut
from app.ml.infer import analyze_xray
from app.ml.preprocessing import validate_image
from app.utils.report_generator import generate_report_pdf

router = APIRouter(prefix="/api/admin", tags=["admin"])


# ---------- Dashboard ----------

@router.get("/dashboard")
def dashboard(db: Session = Depends(get_db), admin: Admin = Depends(require_admin)):
    total_patients = db.query(func.count(Patient.patient_id)).scalar()
    total_analyses = db.query(func.count(AnalysisResult.result_id)).scalar()
    total_reports = db.query(func.count(Report.report_id)).scalar()
    priority_cases = db.query(func.count(AnalysisResult.result_id)).filter(
        AnalysisResult.priority.in_(["Urgent", "High"])
    ).scalar()
    recent = (
        db.query(AnalysisResult)
        .order_by(AnalysisResult.analyzed_at.desc())
        .limit(10)
        .all()
    )
    return {
        "total_patients": total_patients,
        "total_analyses": total_analyses,
        "total_reports": total_reports,
        "priority_cases": priority_cases,
        "recent_analyses": [
            {
                "result_id": r.result_id,
                "top_prediction_label": r.top_prediction_label,
                "top_prediction_confidence": r.top_prediction_confidence,
                "priority": r.priority.value if hasattr(r.priority, "value") else r.priority,
                "analyzed_at": r.analyzed_at,
            }
            for r in recent
        ],
    }


# ---------- Patients ----------

@router.post("/patients", response_model=PatientOut, status_code=201)
def add_patient(payload: PatientCreate, db: Session = Depends(get_db), admin: Admin = Depends(require_admin)):
    try:
        gender = GenderEnum(payload.gender.lower())
    except ValueError:
        raise HTTPException(status_code=400, detail="Gender must be male, female, or other")
    patient = Patient(
        name=payload.name, age=payload.age, gender=gender,
        contact_number=payload.contact_number, address=payload.address,
        registered_by_id=admin.admin_id,
    )
    db.add(patient)
    db.commit()
    db.refresh(patient)
    return patient


@router.get("/patients", response_model=list[PatientOut])
def list_patients(search: str = "", db: Session = Depends(get_db), admin: Admin = Depends(require_admin)):
    query = db.query(Patient)
    if search:
        query = query.filter(Patient.name.ilike(f"%{search}%"))
    return query.order_by(Patient.created_at.desc()).all()


@router.get("/patients/{patient_id}", response_model=PatientOut)
def get_patient(patient_id: str, db: Session = Depends(get_db), admin: Admin = Depends(require_admin)):
    patient = db.query(Patient).filter(Patient.patient_id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient


@router.put("/patients/{patient_id}", response_model=PatientOut)
def update_patient(patient_id: str, payload: PatientUpdate, db: Session = Depends(get_db),
                    admin: Admin = Depends(require_admin)):
    patient = db.query(Patient).filter(Patient.patient_id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    data = payload.dict(exclude_unset=True)
    if "gender" in data:
        data["gender"] = GenderEnum(data["gender"].lower())
    for key, value in data.items():
        setattr(patient, key, value)
    db.commit()
    db.refresh(patient)
    return patient


@router.delete("/patients/{patient_id}", status_code=200)
def delete_patient(patient_id: str, db: Session = Depends(get_db), admin: Admin = Depends(require_admin)):
    patient = db.query(Patient).filter(Patient.patient_id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    import logging
    logger = logging.getLogger(__name__)

    # 1. Manually clean up physical files before DB cascade deletes the records
    for xray in patient.xray_images:
        # For each xray, handle its analyses
        for analysis in xray.analysis_results:
            # Handle report PDF
            if analysis.report and analysis.report.report_path:
                if os.path.exists(analysis.report.report_path):
                    try:
                        os.remove(analysis.report.report_path)
                    except Exception as e:
                        logger.error(f"Failed to delete report file {analysis.report.report_path}: {e}")
            
            # Handle heatmap PNG
            if analysis.heatmap_path:
                # heatmap_path is a URL like /files/uploads/heatmap_xxx.png
                # extract filename and delete from UPLOAD_DIR
                filename = os.path.basename(analysis.heatmap_path)
                local_heatmap_path = os.path.join(settings.UPLOAD_DIR, filename)
                if os.path.exists(local_heatmap_path):
                    try:
                        os.remove(local_heatmap_path)
                    except Exception as e:
                        logger.error(f"Failed to delete heatmap file {local_heatmap_path}: {e}")
        
        # Handle original X-ray image
        if xray.image_path and os.path.exists(xray.image_path):
            try:
                os.remove(xray.image_path)
            except Exception as e:
                logger.error(f"Failed to delete xray file {xray.image_path}: {e}")

    # 2. Delete patient (SQLAlchemy will cascade delete all child records)
    db.delete(patient)
    db.commit()
    
    return {"message": "Patient deleted successfully"}


@router.get("/patients/{patient_id}/xrays")
def list_patient_xrays(patient_id: str, db: Session = Depends(get_db), admin: Admin = Depends(require_admin)):
    patient = db.query(Patient).filter(Patient.patient_id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    xrays = db.query(XrayImage).filter(XrayImage.patient_id == patient_id).order_by(XrayImage.uploaded_at.desc()).all()
    result = []
    for x in xrays:
        analyses = db.query(AnalysisResult).filter(AnalysisResult.image_id == x.image_id).order_by(AnalysisResult.analyzed_at.desc()).all()
        result.append({
            "image_id": x.image_id,
            "uploaded_at": x.uploaded_at,
            "analyses": [
                {
                    "result_id": a.result_id,
                    "top_prediction_label": a.top_prediction_label,
                    "top_prediction_confidence": a.top_prediction_confidence,
                    "priority": a.priority.value if hasattr(a.priority, "value") else a.priority,
                    "is_demo_model": a.is_demo_model,
                    "has_report": a.report is not None,
                    "report_id": a.report.report_id if a.report else None,
                } for a in analyses
            ],
        })
    return result


# ---------- X-ray upload + analysis ----------

@router.post("/patients/{patient_id}/xray")
def upload_xray(patient_id: str, file: UploadFile = File(...), db: Session = Depends(get_db),
                 admin: Admin = Depends(require_admin)):
    patient = db.query(Patient).filter(Patient.patient_id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in settings.ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only JPG/JPEG/PNG files are allowed")

    file.file.seek(0, os.SEEK_END)
    size_mb = file.file.tell() / (1024 * 1024)
    file.file.seek(0)
    if size_mb > settings.MAX_UPLOAD_SIZE_MB:
        raise HTTPException(status_code=400, detail=f"File exceeds {settings.MAX_UPLOAD_SIZE_MB}MB limit")

    saved_filename = f"{uuid.uuid4().hex}{ext}"
    saved_path = os.path.join(settings.UPLOAD_DIR, saved_filename)
    with open(saved_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    if not validate_image(saved_path):
        os.remove(saved_path)
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image")

    xray = XrayImage(
        patient_id=patient_id, image_path=saved_path,
        original_filename=file.filename, uploaded_by_id=admin.admin_id,
    )
    db.add(xray)
    db.commit()
    db.refresh(xray)
    return {"image_id": xray.image_id, "image_path": xray.image_path, "message": "X-ray uploaded successfully"}


@router.get("/xray/{image_id}", response_model=XrayImageOut)
def get_xray_image(image_id: str, db: Session = Depends(get_db), admin: Admin = Depends(require_admin)):
    xray = db.query(XrayImage).filter(XrayImage.image_id == image_id).first()
    if not xray:
        raise HTTPException(status_code=404, detail="X-ray image not found")
    return xray


@router.post("/xray/{image_id}/analyze", response_model=AnalysisResultOut)
def start_analysis(image_id: str, db: Session = Depends(get_db), admin: Admin = Depends(require_admin)):
    xray = db.query(XrayImage).filter(XrayImage.image_id == image_id).first()
    if not xray:
        raise HTTPException(status_code=404, detail="X-ray image not found")

    result = analyze_xray(xray.image_path)

    analysis = AnalysisResult(
        image_id=image_id,
        predicted_conditions=json.dumps(result["predicted_conditions"]),
        top_prediction_label=result["top_prediction_label"],
        top_prediction_confidence=result["top_prediction_confidence"],
        uncertainty_status=result["uncertainty_status"],
        priority=result["priority"],
        heatmap_path=result["heatmap_path"],
        model_version=result["model_version"],
        is_demo_model=result["is_demo_model"],
    )
    db.add(analysis)
    db.commit()
    db.refresh(analysis)

    return AnalysisResultOut(
        result_id=analysis.result_id,
        image_id=analysis.image_id,
        predicted_conditions=result["predicted_conditions"],
        top_prediction_label=analysis.top_prediction_label,
        top_prediction_confidence=analysis.top_prediction_confidence,
        uncertainty_status=analysis.uncertainty_status.value if hasattr(analysis.uncertainty_status, "value") else analysis.uncertainty_status,
        priority=analysis.priority.value if hasattr(analysis.priority, "value") else analysis.priority,
        heatmap_path=analysis.heatmap_path,
        is_demo_model=analysis.is_demo_model,
        disclaimer=result["disclaimer"],
        analyzed_at=analysis.analyzed_at,
    )


@router.get("/analysis/{result_id}", response_model=AnalysisResultOut)
def get_analysis_result(result_id: str, db: Session = Depends(get_db), admin: Admin = Depends(require_admin)):
    r = db.query(AnalysisResult).filter(AnalysisResult.result_id == result_id).first()
    if not r:
        raise HTTPException(status_code=404, detail="Analysis result not found")
    return AnalysisResultOut(
        result_id=r.result_id, image_id=r.image_id,
        predicted_conditions=json.loads(r.predicted_conditions),
        top_prediction_label=r.top_prediction_label,
        top_prediction_confidence=r.top_prediction_confidence,
        uncertainty_status=r.uncertainty_status.value if hasattr(r.uncertainty_status, "value") else r.uncertainty_status,
        priority=r.priority.value if hasattr(r.priority, "value") else r.priority,
        heatmap_path=r.heatmap_path,
        is_demo_model=r.is_demo_model,
        disclaimer=(
            "DEMO MODE prediction - not clinically meaningful." if r.is_demo_model
            else "Academic prototype - not clinically validated."
        ),
        analyzed_at=r.analyzed_at,
    )


@router.get("/analysis-history", response_model=list[AnalysisResultOut])
def analysis_history(db: Session = Depends(get_db), admin: Admin = Depends(require_admin)):
    results = db.query(AnalysisResult).order_by(AnalysisResult.analyzed_at.desc()).all()
    return [
        AnalysisResultOut(
            result_id=r.result_id, image_id=r.image_id,
            predicted_conditions=json.loads(r.predicted_conditions),
            top_prediction_label=r.top_prediction_label,
            top_prediction_confidence=r.top_prediction_confidence,
            uncertainty_status=r.uncertainty_status.value if hasattr(r.uncertainty_status, "value") else r.uncertainty_status,
            priority=r.priority.value if hasattr(r.priority, "value") else r.priority,
            heatmap_path=r.heatmap_path,
            is_demo_model=r.is_demo_model,
            disclaimer=(
                "DEMO MODE prediction - not clinically meaningful." if r.is_demo_model
                else "Academic prototype - not clinically validated."
            ),
            analyzed_at=r.analyzed_at,
        ) for r in results
    ]


# ---------- Reports ----------

@router.post("/analysis/{result_id}/report", response_model=ReportOut, status_code=201)
def generate_report(result_id: str, db: Session = Depends(get_db), admin: Admin = Depends(require_admin)):
    analysis = db.query(AnalysisResult).filter(AnalysisResult.result_id == result_id).first()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis result not found")
    xray = db.query(XrayImage).filter(XrayImage.image_id == analysis.image_id).first()
    patient = db.query(Patient).filter(Patient.patient_id == xray.patient_id).first()

    report_filename = f"report_{uuid.uuid4().hex}.pdf"
    report_path = os.path.join(settings.REPORT_DIR, report_filename)
    generate_report_pdf(patient, xray, analysis, report_path)

    report = Report(analysis_result_id=result_id, report_path=report_path, generated_by_id=admin.admin_id)
    db.add(report)
    db.commit()
    db.refresh(report)
    return report


@router.get("/reports", response_model=list[ReportOut])
def list_reports(db: Session = Depends(get_db), admin: Admin = Depends(require_admin)):
    return db.query(Report).order_by(Report.generated_at.desc()).all()


@router.get("/reports/{report_id}")
def get_report_detail(report_id: str, db: Session = Depends(get_db), admin: Admin = Depends(require_admin)):
    report = db.query(Report).filter(Report.report_id == report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    analysis = db.query(AnalysisResult).filter(AnalysisResult.result_id == report.analysis_result_id).first()
    xray = db.query(XrayImage).filter(XrayImage.image_id == analysis.image_id).first()
    patient = db.query(Patient).filter(Patient.patient_id == xray.patient_id).first()

    return {
        "report_id": report.report_id,
        "report_path": report.report_path,
        "generated_at": report.generated_at,
        "patient": PatientOut.model_validate(patient),
        "xray_image_path": xray.image_path,
        "analysis": AnalysisResultOut(
            result_id=analysis.result_id, image_id=analysis.image_id,
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
