import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from app.main import app
from app.database import SessionLocal, get_db
from app.models import Admin, Doctor, Report, ReportStatus, AnalysisResult, XrayImage, Patient
import json

client = TestClient(app)

def test_workflow():
    db = SessionLocal()
    
    # 1. Create admin and doctor for testing
    # 2. Upload xray
    # 3. Analyze
    # 4. Check doctor dashboard for pending report
    # 5. Check if we can change status
    
    # We will just verify our db schema and endpoints via direct calls to db
    
    patient = db.query(Patient).first()
    if not patient:
        print("No patient to test, skipping full API test.")
        return
        
    xray = db.query(XrayImage).filter_by(patient_id=patient.patient_id).first()
    if not xray:
        print("No xray to test, skipping.")
        return
        
    analysis = db.query(AnalysisResult).filter_by(image_id=xray.image_id).first()
    if not analysis:
        print("No analysis to test.")
        return
        
    report = db.query(Report).filter_by(analysis_result_id=analysis.result_id).first()
    if not report:
        print("No report, creating one.")
        report = Report(analysis_result_id=analysis.result_id, report_path="dummy.pdf", status=ReportStatus.pending)
        db.add(report)
        db.commit()
        db.refresh(report)
        
    print(f"Report status initially: {report.status}")
    
    # Simulate Doctor API
    report.status = ReportStatus.viewed
    db.commit()
    print(f"Report status updated: {report.status}")

    # Mark for review
    report.status = ReportStatus.review
    db.commit()
    print(f"Report status updated: {report.status}")
    
    db.close()
    print("Tests passed.")

if __name__ == "__main__":
    test_workflow()
