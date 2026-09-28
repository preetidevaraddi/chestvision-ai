import os
from collections import namedtuple
from app.utils.report_generator import generate_report_pdf

def test_pdf():
    Patient = namedtuple('Patient', ['patient_id', 'name', 'age', 'gender'])
    XrayImage = namedtuple('XrayImage', ['image_id', 'uploaded_at', 'image_path'])
    AnalysisResult = namedtuple('AnalysisResult', ['predicted_conditions', 'top_prediction_label', 
                                                   'top_prediction_confidence', 'uncertainty_status', 
                                                   'priority', 'heatmap_path', 'is_demo_model'])
                                                   
    patient = Patient(patient_id="P1", name="Test", age=30, gender="Male")
    
    import datetime
    
    heatmap_actual = r"C:\Users\Preeti Devaraddi\Documents\chestvision-ai-copy\backend\uploads\heatmap_954d4a678a0e461c9363e2c7ee2cdf90.png"
    
    xray = XrayImage(image_id="X1", uploaded_at=datetime.datetime.now(), 
                     image_path=heatmap_actual)
                     
    import json
    cond = json.dumps([{"label": "Tuberculosis", "confidence": 0.9}])
    analysis = AnalysisResult(predicted_conditions=cond, top_prediction_label="Tuberculosis",
                              top_prediction_confidence=0.9, uncertainty_status="Clear", priority="High",
                              heatmap_path=heatmap_actual,
                              is_demo_model=False)
                              
    out = "test_report2.pdf"
    generate_report_pdf(patient, xray, analysis, out)
    print(f"Report generated: {out}")
    print(f"File size: {os.path.getsize(out)}")

if __name__ == '__main__':
    test_pdf()
