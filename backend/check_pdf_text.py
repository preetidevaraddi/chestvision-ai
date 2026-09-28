with open("test_report2.pdf", "rb") as f:
    content = f.read()
    if b"Visual Explanation" in content:
        print("Visual Explanation text found in PDF binary")
    else:
        print("Text NOT found in PDF binary")
