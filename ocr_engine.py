import cv2
import pytesseract
import numpy as np
import easyocr
import re

# ==============================
# TESSERACT PATH (WINDOWS)
# ==============================
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ==============================
# LOAD EASYOCR (ONCE)
# ==============================
reader = easyocr.Reader(['en'], gpu=False)


# ==============================
# PRINTED TEXT PREPROCESSING
# ==============================
def preprocess_printed(image_path):
    img = cv2.imread(image_path)

    img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)

    # Sharpen
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    sharp = cv2.filter2D(gray, -1, kernel)

    thresh = cv2.adaptiveThreshold(
        sharp, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )

    return thresh


# ==============================
# HANDWRITTEN PREPROCESSING
# ==============================
def preprocess_handwritten(image_path):
    img = cv2.imread(image_path)

    # Bigger resize for handwriting
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Smooth + denoise
    gray = cv2.GaussianBlur(gray, (5,5), 0)

    # Adaptive threshold
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )

    return thresh


# ==============================
# CLEAN TEXT
# ==============================
def clean_text(text):
    text = text.replace("|", "I")

    # Keep line structure
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n+', '\n', text)

    return text.strip()


# ==============================
# HANDWRITTEN OCR (EASYOCR)
# ==============================
def extract_handwritten(image_path):
    processed = preprocess_handwritten(image_path)

    results = reader.readtext(
        processed,
        detail=0,
        paragraph=True,
        contrast_ths=0.05,
        adjust_contrast=0.7
    )

    text = "\n".join(results)

    if len(text.strip()) < 5:
        return "⚠️ Could not clearly recognize handwriting"

    return clean_text(text)


# ==============================
# PRINTED OCR (TESSERACT)
# ==============================
def extract_printed(image_path):
    processed = preprocess_printed(image_path)

    custom_config = r'--oem 3 --psm 4'

    text = pytesseract.image_to_string(processed, config=custom_config)

    return clean_text(text)


# ==============================
# MAIN OCR FUNCTION (HYBRID)
# ==============================
def extract_text(image_path, doc_type="printed"):

    if doc_type == "handwritten":
        return extract_handwritten(image_path)

    return extract_printed(image_path)