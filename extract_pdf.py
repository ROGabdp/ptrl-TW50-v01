from pypdf import PdfReader

pdf_path = "1-s2.0-S0957417424013319-main.pdf"
try:
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    
    with open("pdf_content.txt", "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Successfully extracted {len(text)} characters to pdf_content.txt")
except Exception as e:
    print(f"Error extracting text: {e}")
