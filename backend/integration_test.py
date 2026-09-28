import os
from fastapi.testclient import TestClient
from app.main import app
from app.database import Base, engine, SessionLocal
from app.core.security import get_password_hash
from app.models import Admin, Doctor, Patient, XRayImage

# Initialize test DB
Base.metadata.drop_all(bind=engine)
Base.metadata.create_all(bind=engine)

# Seed Doctor
db = SessionLocal()
doctor = Doctor(email="doc@test.com", password_hash=get_password_hash("pass"), full_name="Doc", specialty="Test", license_number="123")
db.add(doctor)
db.commit()
db.refresh(doctor)

# Seed Patient
patient = Patient(doctor_id=doctor.id, first_name="John", last_name="Doe", date_of_birth="1980-01-01", gender="M", medical_history="")
db.add(patient)
db.commit()
db.refresh(patient)
db.close()

client = TestClient(app)

# Login
login_res = client.post("/doctor/login", json={"email": "doc@test.com", "password": "pass"})
token = login_res.json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}

# Test 1: Upload X-Ray and Analyze (Pneumonia)
with open("dataset/images/Pneumonia_0001.jpeg", "rb") as f:
    upload_res = client.post(f"/patients/{patient.id}/xray", headers=headers, files={"file": ("test.jpeg", f, "image/jpeg")})
image_id = upload_res.json()["id"]

# Analyze
analyze_res = client.post(f"/xray/{image_id}/analyze", headers=headers)
data = analyze_res.json()

print("\n=== FASTAPI ENDPOINT INTEGRATION TEST ===")
print("Status Code:", analyze_res.status_code)
print("Top Prediction:", data["top_prediction_label"])
print("Model Version:", data["model_version"])
print("Disclaimer:", data["disclaimer"])
print("Grad-CAM Path:", data["heatmap_path"])

sum_p = sum(c["confidence"] for c in data["predicted_conditions"])
print("Sum of Probabilities:", sum_p)

# Test 2: Upload X-Ray and Analyze (Normal)
with open("dataset/images/Normal_0001.png", "rb") as f:
    upload_res2 = client.post(f"/patients/{patient.id}/xray", headers=headers, files={"file": ("test2.png", f, "image/png")})
image_id2 = upload_res2.json()["id"]

analyze_res2 = client.post(f"/xray/{image_id2}/analyze", headers=headers)
data2 = analyze_res2.json()

print("\n--- TEST 2 (Normal) ---")
print("Top Prediction:", data2["top_prediction_label"])
sum_p2 = sum(c["confidence"] for c in data2["predicted_conditions"])
print("Sum of Probabilities:", sum_p2)
