import os
import io
from fastapi.testclient import TestClient
from app.main import app
from app.database import get_db
from app.models import Admin, Patient, XrayImage, AnalysisResult, Report

client = TestClient(app)

def override_require_admin():
    db = next(get_db())
    admin = db.query(Admin).first()
    return admin

from app.dependencies import require_admin
app.dependency_overrides[require_admin] = override_require_admin

def main():
    print("--- 1. Set up Test Patient ---")
    resp = client.post("/api/admin/patients", json={
        "name": "Delete Test Patient",
        "age": 30,
        "gender": "female",
        "contact_number": "123-456",
        "address": "Delete Ave"
    })
    patient_id = resp.json()["patient_id"]
    print(f"Created patient {patient_id}")
    
    # Upload X-ray
    test_img_path = 'dataset/real_images/Normal_0000.png'
    with open(test_img_path, "rb") as f:
        resp = client.post(
            f"/api/admin/patients/{patient_id}/xray",
            files={"file": (os.path.basename(test_img_path), f, "image/png")}
        )
    image_id = resp.json()["image_id"]
    image_path = resp.json()["image_path"]
    
    # Analyze
    resp = client.post(f"/api/admin/xray/{image_id}/analyze")
    analysis_data = resp.json()
    result_id = analysis_data["result_id"]
    heatmap_url = analysis_data["heatmap_path"]
    heatmap_path = os.path.join("uploads", os.path.basename(heatmap_url))
    
    # Report
    resp = client.post(f"/api/admin/analysis/{result_id}/report")
    report_path = resp.json()["report_path"]
    
    print(f"File created: {image_path}")
    print(f"File created: {heatmap_path}")
    print(f"File created: {report_path}")
    
    assert os.path.exists(image_path)
    assert os.path.exists(heatmap_path)
    assert os.path.exists(report_path)
    
    print("--- 2. Execute DELETE API ---")
    resp = client.delete(f"/api/admin/patients/{patient_id}")
    assert resp.status_code == 200
    print("DELETE returned 200 OK")
    
    print("--- 3. Verify Files and Database ---")
    assert not os.path.exists(image_path), "X-ray file was NOT deleted!"
    assert not os.path.exists(heatmap_path), "Heatmap file was NOT deleted!"
    assert not os.path.exists(report_path), "Report file was NOT deleted!"
    print("All physical files successfully deleted.")
    
    # DB check
    db = next(get_db())
    assert db.query(Patient).filter(Patient.patient_id == patient_id).first() is None
    assert db.query(XrayImage).filter(XrayImage.patient_id == patient_id).first() is None
    assert db.query(AnalysisResult).filter(AnalysisResult.image_id == image_id).first() is None
    assert db.query(Report).filter(Report.analysis_result_id == result_id).first() is None
    print("All database records successfully cascaded and deleted.")
    
    print("TEST PASSED SUCCESSFULLY")

if __name__ == '__main__':
    main()
