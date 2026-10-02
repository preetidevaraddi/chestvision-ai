import os
import random
import sqlalchemy
from sqlalchemy import text
from app.database import engine

def upgrade():
    with engine.connect() as conn:
        try:
            conn.execute(text("ALTER TABLE patients ADD COLUMN patient_display_id VARCHAR(10);"))
        except Exception as e:
            print(f"patients col exists: {e}")
            
        try:
            conn.execute(text("ALTER TABLE analysis_results ADD COLUMN result_display_id VARCHAR(10);"))
        except Exception as e:
            print(f"analysis_results col exists: {e}")
            
        try:
            conn.execute(text("ALTER TABLE reports ADD COLUMN report_display_id VARCHAR(10);"))
        except Exception as e:
            print(f"reports col exists: {e}")
            
        # Assign random IDs to existing rows
        patients = conn.execute(text("SELECT patient_id FROM patients WHERE patient_display_id IS NULL")).fetchall()
        for p in patients:
            did = str(random.randint(10000, 99999))
            conn.execute(text("UPDATE patients SET patient_display_id = :did WHERE patient_id = :pid"), {"did": did, "pid": p[0]})
            
        results = conn.execute(text("SELECT result_id FROM analysis_results WHERE result_display_id IS NULL")).fetchall()
        for r in results:
            did = str(random.randint(10000, 99999))
            conn.execute(text("UPDATE analysis_results SET result_display_id = :did WHERE result_id = :rid"), {"did": did, "rid": r[0]})
            
        reports = conn.execute(text("SELECT report_id FROM reports WHERE report_display_id IS NULL")).fetchall()
        for r in reports:
            did = str(random.randint(10000, 99999))
            conn.execute(text("UPDATE reports SET report_display_id = :did WHERE report_id = :rid"), {"did": did, "rid": r[0]})
            
        conn.commit()
        print("Database updated successfully.")

if __name__ == "__main__":
    upgrade()
