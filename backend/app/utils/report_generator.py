"""
Generates the medical-style AI analysis report PDF (spec section 17).
"""
import json
import os
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

from app.core.config import settings

DISCLAIMER = (
    "This AI-assisted report is intended for decision support and should not be "
    "considered a final medical diagnosis or a replacement for professional "
    "medical evaluation."
)


def generate_report_pdf(patient, xray_image, analysis_result, output_path: str) -> str:
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleStyle", parent=styles["Title"], textColor=colors.HexColor("#0f4c81"))
    heading_style = ParagraphStyle("Heading", parent=styles["Heading2"], textColor=colors.HexColor("#0f4c81"))
    normal = styles["Normal"]

    doc = SimpleDocTemplate(output_path, pagesize=A4,
                             topMargin=20 * mm, bottomMargin=20 * mm)
    story = []

    story.append(Paragraph("ChestVision AI", title_style))
    story.append(Paragraph("AI-Assisted Chest X-Ray Analysis Report", normal))
    story.append(Spacer(1, 6))

    if getattr(analysis_result, "is_demo_model", True):
        demo_style = ParagraphStyle("DemoBanner", parent=normal, textColor=colors.white,
                                     backColor=colors.HexColor("#c0392b"), fontSize=10,
                                     spaceBefore=4, spaceAfter=4, leftIndent=4)
        story.append(Paragraph(
            "&nbsp;DEMO MODE - no trained model loaded. This prediction is NOT clinically "
            "meaningful and is shown for interface demonstration only.&nbsp;", demo_style))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Report Information", heading_style))
    story.append(Paragraph(f"Report generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", normal))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Patient Information", heading_style))
    patient_table = Table([
        ["Patient ID", patient.patient_id],
        ["Name", patient.name],
        ["Age", str(patient.age)],
        ["Gender", patient.gender.value if hasattr(patient.gender, "value") else str(patient.gender)],
    ], colWidths=[120, 300])
    patient_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eaf2f8")),
    ]))
    story.append(patient_table)
    story.append(Spacer(1, 8))

    story.append(Paragraph("X-Ray Information", heading_style))
    story.append(Paragraph(f"Image reference: {xray_image.image_id}", normal))
    story.append(Paragraph(f"Uploaded: {xray_image.uploaded_at.strftime('%Y-%m-%d %H:%M UTC')}", normal))
    story.append(Spacer(1, 8))

    story.append(Paragraph("AI Analysis", heading_style))
    conditions = json.loads(analysis_result.predicted_conditions)
    cond_rows = [["Condition", "Confidence"]] + [
        [c["label"], f"{c['confidence']*100:.1f}%"] for c in conditions[:6]
    ]
    cond_table = Table(cond_rows, colWidths=[300, 120])
    cond_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f4c81")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ]))
    story.append(cond_table)
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        f"Uncertainty status: <b>{analysis_result.uncertainty_status.value if hasattr(analysis_result.uncertainty_status, 'value') else analysis_result.uncertainty_status}</b>",
        normal))
    story.append(Spacer(1, 8))

    if xray_image.image_path and os.path.exists(xray_image.image_path):
        story.append(Paragraph("Visual Explanation", heading_style))
        images_row = []
        try:
            images_row.append(RLImage(xray_image.image_path, width=180, height=180))
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Failed to embed X-ray image: {e}")
            
        if analysis_result.heatmap_path:
            # heatmap_path from DB is now a URL like /files/uploads/heatmap_xxx.png
            # Convert it back to local filesystem path for ReportLab
            filename = os.path.basename(analysis_result.heatmap_path)
            local_heatmap_path = os.path.join(settings.UPLOAD_DIR, filename)
            
            if os.path.exists(local_heatmap_path):
                try:
                    images_row.append(RLImage(local_heatmap_path, width=180, height=180))
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).error(f"Failed to embed heatmap image: {e}")
                    
        if images_row:
            story.append(Table([images_row]))
        story.append(Paragraph(
            "Note: highlighted regions indicate areas that influenced the AI "
            "prediction and are not a confirmed disease location.", normal))
        story.append(Spacer(1, 8))

    story.append(Paragraph("Priority", heading_style))
    priority_val = analysis_result.priority.value if hasattr(analysis_result.priority, "value") else analysis_result.priority
    story.append(Paragraph(f"Priority level: <b>{priority_val}</b>", normal))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Summary", heading_style))
    top_label = analysis_result.top_prediction_label
    top_conf = analysis_result.top_prediction_confidence * 100
    story.append(Paragraph(
        f"The AI analysis suggests findings most consistent with <b>{top_label}</b> "
        f"({top_conf:.1f}% confidence). Priority classified as {priority_val}.", normal))
    story.append(Spacer(1, 14))

    disclaimer_style = ParagraphStyle("Disclaimer", parent=normal, textColor=colors.HexColor("#b03a2e"),
                                       fontSize=9, italic=True)
    story.append(Paragraph(f"<i>{DISCLAIMER}</i>", disclaimer_style))

    doc.build(story)
    return output_path
