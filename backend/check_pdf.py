import PyPDF2

def count_images_in_pdf(pdf_path):
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        count = 0
        for page in reader.pages:
            if '/XObject' in page['/Resources']:
                xObject = page['/Resources']['/XObject'].get_object()
                for obj in xObject:
                    if xObject[obj]['/Subtype'] == '/Image':
                        count += 1
        return count

if __name__ == '__main__':
    print(f"Images in PDF: {count_images_in_pdf('test_report.pdf')}")
