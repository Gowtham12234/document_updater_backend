# server/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import shutil
# app.py (near imports)
import nltk

# Try to download minimal NLTK data required for summarization.
# For development only — in production pre-download data or vendor a copy.
nltk_resources = ['stopwords', 'punkt']
for res in nltk_resources:
    try:
        nltk.data.find(f'corpora/{res}')
        print(f"[nltk] resource '{res}' already available.")
    except LookupError:
        print(f"[nltk] resource '{res}' not found. Downloading...")
        nltk.download(res, quiet=True)
        print(f"[nltk] downloaded '{res}'.")

from processors.extraction_service import extract_text
from processors.summary_service import summarize_text

app = Flask(__name__)
# Enable CORS for communication with React frontend
CORS(app) 

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if os.path.exists(UPLOAD_FOLDER):
    shutil.rmtree(UPLOAD_FOLDER) # Clear previous uploads on start
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload_and_summarize', methods=['POST'])
def upload_and_summarize():
    # 1. Check for file upload 
    if 'document' not in request.files:
        return jsonify({"message": "No file part"}), 400
        
    file = request.files['document']
    summary_length = request.form.get('length', 'medium')

    if file.filename == '':
        return jsonify({"message": "No selected file"}), 400

    if file and allowed_file(file.filename):
        # 2. Save the file temporarily
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # 3. Extract Text [cite: 68]
        mime_type = file.content_type
        raw_text = extract_text(file_path, mime_type)
        
        # 4. Generate Summary 
        summary = summarize_text(raw_text, summary_length)
        
        # 5. Clean up the uploaded file
        os.remove(file_path)

        # 6. Return results
        if "Error" in raw_text or "Cannot summarize" in summary:
             return jsonify({
                "raw_text": raw_text,
                "summary": "Summary could not be generated.",
                "message": raw_text
             }), 500
        
        return jsonify({
            "summary": summary,
            "raw_text": raw_text,
            "message": f"Successfully summarized {filename}."
        })
    else:
        return jsonify({"message": "File type not allowed. Must be PDF, JPG, or PNG."}), 400

if __name__ == '__main__':
    # Use host='0.0.0.0' to be accessible by your React app
    app.run(debug=True, host='0.0.0.0', port=5000)