import os
import io
from fastapi.testclient import TestClient
from app.main import app
from app.database import get_db, Base, engine
from sqlalchemy.orm import sessionmaker
from app.models import Admin

# Setup test DB (or use real one, doesn't matter for read/write if we clean up or just leave it)
client = TestClient(app)

def override_require_admin():
    db = next(get_db())
    admin = db.query(Admin).first()
    return admin

from app.dependencies import require_admin
app.dependency_overrides[require_admin] = override_require_admin

def main():
    print("1. Locating test images...")
    dataset_dir = 'dataset/real_images'
    valid_images = []
    
    if os.path.exists(dataset_dir):
        files = sorted(os.listdir(dataset_dir))
        for prefix in ['Normal', 'Pneumonia', 'Tuberculosis', 'COVID19', 'Nodule']:
            for f in files:
                if f.startswith(prefix) and (f.endswith('.png') or f.endswith('.jpeg')):
                    valid_images.append(os.path.join(dataset_dir, f))
                    break
                    
    valid_images = valid_images[:3] # Just grab 3
    
    if len(valid_images) < 3:
        print(f"Could not find 3 valid test images. Found: {valid_images}")
        return
        
    print("2. Creating test patient...")
    resp = client.post("/api/admin/patients", json={
        "name": "Test Workflow Patient",
        "age": 45,
        "gender": "male",
        "contact_number": "555-0199",
        "address": "123 Test St"
    })
    patient_id = resp.json()["patient_id"]
    
    last_result_id = None
    last_report_path = None
    
    for i, test_img_path in enumerate(valid_images):
        print(f"\n--- Testing Image {i+1}: {test_img_path} ---")
        
        with open(test_img_path, "rb") as f:
            resp = client.post(
                f"/api/admin/patients/{patient_id}/xray",
                files={"file": (os.path.basename(test_img_path), f, "image/png")}
            )
        assert resp.status_code == 200
        image_id = resp.json()["image_id"]
        
        print("Analyzing X-ray...")
        resp = client.post(f"/api/admin/xray/{image_id}/analyze")
        assert resp.status_code == 200
        analysis_data = resp.json()
        result_id = analysis_data["result_id"]
        heatmap_url = analysis_data["heatmap_path"]
        print(f"Heatmap URL returned by API: {heatmap_url}")
        
        resp = client.get(heatmap_url)
        assert resp.status_code == 200
        print(f"Heatmap image successfully served over HTTP! Size: {len(resp.content)} bytes")
        
        last_result_id = result_id
        
    print("\n--- Generating Final Report ---")
    resp = client.post(f"/api/admin/analysis/{last_result_id}/report")
    assert resp.status_code == 201
    report_data = resp.json()
    report_path = report_data["report_path"]
    
    print(f"Verifying PDF images in {report_path}...")
    assert os.path.exists(report_path)
    
    file_size = os.path.getsize(report_path)
    print(f"PDF File Size: {file_size} bytes")
    if file_size > 300000:
        print("SUCCESS! File size indicates both original X-ray and Grad-CAM overlay are embedded (>300KB).")
    else:
        print("FAILURE! File size is too small, one or both images are missing.")
        
if __name__ == '__main__':
    main()
