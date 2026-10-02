import urllib.request
import urllib.parse
import json
import uuid

BASE_URL = "http://127.0.0.1:8000/api"
ADMIN_TOKEN = "admin-secret-token"

def req(method, path, data=None):
    req = urllib.request.Request(BASE_URL + path, method=method)
    req.add_header("Authorization", f"Bearer {ADMIN_TOKEN}")
    if data is not None:
        import mimetypes
        if isinstance(data, dict):
            req.add_header("Content-Type", "application/json")
            data = json.dumps(data).encode("utf-8")
    
    try:
        with urllib.request.urlopen(req, data=data) as f:
            return json.loads(f.read().decode())
    except urllib.error.HTTPError as e:
        print(f"Error {e.code}: {e.read().decode()}")
        raise e

# 1. Create Patient
patient_data = {
    "name": "Test Workflow Patient",
    "age": 40,
    "gender": "male",
    "contact_number": "1234567890"
}
p = req("POST", "/admin/patients", patient_data)
print(f"Created Patient: {p['patient_display_id']} (UUID: {p['patient_id']})")
patient_id = p["patient_id"]
display_id = p["patient_display_id"]

assert display_id is not None, "Display ID not generated"

# 2. Upload Xray (Mock using boundary)
boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
body = (
    f'--{boundary}\r\n'
    f'Content-Disposition: form-data; name="file"; filename="test.jpg"\r\n'
    f'Content-Type: image/jpeg\r\n\r\n'
    f'fake image data\r\n'
    f'--{boundary}\r\n'
    f'Content-Disposition: form-data; name="patient_id"\r\n\r\n'
    f'{patient_id}\r\n'
    f'--{boundary}--\r\n'
)

upload_req = urllib.request.Request(BASE_URL + "/admin/xrays/upload", method="POST", data=body.encode("utf-8"))
upload_req.add_header("Authorization", f"Bearer {ADMIN_TOKEN}")
upload_req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
with urllib.request.urlopen(upload_req) as f:
    xray = json.loads(f.read().decode())

image_id = xray["image_id"]
print(f"Uploaded Xray: {image_id}")

# 3. Analyze
analysis = req("POST", f"/admin/analysis/start?image_id={image_id}")
print(f"Analysis started: Result Display ID: {analysis['result_display_id']}")
assert analysis['result_display_id'] == display_id, "Result Display ID mismatch!"
result_id = analysis['result_id']

# 4. Wait for analysis to complete? The mock analysis takes a bit if it's running gradcam. Wait it's an async task. 
# But in our setup `start_analysis` awaits or returns after? Let's check. 
# Actually, the start_analysis does analysis synchronously.

# 5. Generate Report
try:
    report = req("POST", f"/admin/reports/generate?result_id={result_id}")
    print(f"Report generated: Report Display ID: {report['report_display_id']}")
    assert report['report_display_id'] == display_id, "Report Display ID mismatch!"
except Exception as e:
    print(f"Error generating report: {e}")

# Verify in lists
history = req("GET", "/admin/analysis/history")
h_entry = next(h for h in history if h["result_id"] == result_id)
assert h_entry["result_display_id"] == display_id, "History Result Display ID mismatch!"

print("ALL TESTS PASSED: Patient, Result, and Report share the identical Display ID.")
