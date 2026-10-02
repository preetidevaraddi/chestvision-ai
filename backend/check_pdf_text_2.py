import PyPDF2

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text

if __name__ == '__main__':
    text = extract_text_from_pdf('test_report2.pdf')
    print("PDF Text Content:")
    print("=" * 40)
    print(text)
    print("=" * 40)
    
    assert "P1" in text, "Patient ID not found"
    assert "Summary" not in text, "Summary section should be removed"
    assert "disclaimer" not in text.lower(), "Disclaimer should be removed"
    assert "Image reference" not in text, "Image reference should be removed"
    assert "AI-highlighted regions indicate areas of the X-ray that contributed to the model's prediction." in text, "Visual explanation text is wrong"
    print("All tests passed!")
