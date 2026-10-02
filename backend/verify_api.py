import urllib.request
import json
import uuid

BASE_URL = "http://127.0.0.1:8000/api"
ADMIN_TOKEN = "admin-secret-token"

def req(method, path, data=None):
    request = urllib.request.Request(BASE_URL + path, method=method)
    request.add_header("Authorization", f"Bearer {ADMIN_TOKEN}")
    if data is not None:
        request.add_header("Content-Type", "application/json")
        data = json.dumps(data).encode("utf-8")
    
    with urllib.request.urlopen(request, data=data) as f:
        return json.loads(f.read().decode())

try:
    dashboard = req("GET", "/admin/dashboard")
    print("Dashboard:", dashboard.keys())
    if dashboard["recent_analyses"]:
        a = dashboard["recent_analyses"][0]
        print("Analysis keys:", a.keys())
    
    reports = req("GET", "/admin/reports")
    if reports:
        print("Report keys:", reports[0].keys())
        
    print("All good!")
except Exception as e:
    print("Error:", e)
