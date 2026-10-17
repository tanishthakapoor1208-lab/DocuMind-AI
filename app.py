from flask import Flask, render_template, request, jsonify
import os
import time
from PIL import Image
import pytesseract
import easyocr
import re

from cnn.cnn_classifier import predict_image

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Tesseract
if os.name == "nt":
    pytesseract.pytesseract.tesseract_cmd = (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )
else:
    pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract""

# EasyOCR
reader = easyocr.Reader(['en'])

# ---------------- OCR ----------------
def extract_text(image_path):
    try:
        result = reader.readtext(image_path, detail=0)
        easy_text = "\n".join(result)

        img = Image.open(image_path)
        tess_text = pytesseract.image_to_string(img)

        return (easy_text + "\n" + tess_text).strip()

    except Exception as e:
        print("OCR Error:", e)
        return "OCR Failed"


# ---------------- CLEAN ----------------
def clean_text(text):
    return "\n".join([l.strip() for l in text.split("\n") if len(l.strip()) > 2])


# ---------------- FIELDS ----------------
def extract_fields(text):
    date_match = re.search(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', text)
    amount_match = re.search(r'₹?\s?\d+(?:,\d{3})*(?:\.\d{2})?', text)
    email_match = re.search(r'\S+@\S+', text)

    return {
        "Date": date_match.group() if date_match else None,
        "Amount": amount_match.group() if amount_match else None,
        "Email": email_match.group() if email_match else None
    }


# ---------------- ENTITIES ----------------
def extract_entities(text):
    return {
        "dates": re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', text),
        "money": re.findall(r'₹?\s?\d+(?:,\d{3})*(?:\.\d{2})?', text),
        "org": [],
        "person": re.findall(r'\S+@\S+', text)
    }


# ---------------- ROUTES ----------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")

    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    start_time = time.time()

    try:
        path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(path)

        # OCR
        raw_text = extract_text(path)
        text = clean_text(raw_text)

        # CNN
        cnn_result = predict_image(path)
        doc_type = cnn_result["class"]
        confidence = cnn_result["confidence"]

        # Fields
        fields = extract_fields(text)

        # Entities
        entities = extract_entities(text)

        end_time = time.time()
        total_time = round(end_time - start_time, 2)

        return jsonify({
            "text": text,
            "document_type": doc_type,
            "confidence": confidence,
            "fields": fields,
            "entities": entities,
            "time": f"{total_time}s"
        })

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": "Processing failed"}), 500



if __name__ == "__main__":
    app.run(debug=True)
