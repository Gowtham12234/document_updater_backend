# server/processors/extraction_service.py
import PyPDF2
from PIL import Image
import pytesseract
import os

# Set Tesseract path if needed (e.g., if you installed it manually on Windows)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text(file_path, mime_type):
    """
    Extracts text from PDF or image files.
    """
    try:
        if 'pdf' in mime_type:
            # PDF Parsing [cite: 69]
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
                
        elif 'image' in mime_type or os.path.splitext(file_path)[1] in ('.jpg', '.jpeg', '.png'):
            # OCR for Images [cite: 70]
            text = pytesseract.image_to_string(Image.open(file_path))
            return text
            
        else:
            return "Error: Unsupported file type."
            
    except Exception as e:
        print(f"Extraction Error: {e}")
        return f"Extraction failed: {str(e)}"