import os
import re
import cv2
import json
import tempfile
import numpy as np
from PIL import Image
from pdf2image import convert_from_path
import pytesseract
from sklearn.cluster import KMeans
import spacy

# -----------------------------
# Image Preprocessing
# -----------------------------
def preprocess_image(img_path, save_path=None):
    """Preprocess an image for OCR (grayscale, denoise, sharpen)."""
    image = cv2.imread(img_path)
    if image is None:
        raise ValueError(f"Could not read image at {img_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if save_path:
        cv2.imwrite(save_path, gray)

    return gray


# -----------------------------
# OCR
# -----------------------------
def ocr_image(img_path, save_path=None):
    """Run Tesseract OCR and return/save tokens."""
    img = Image.open(img_path)
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

    tokens = []
    for i in range(len(data['text'])):
        if data['text'][i].strip():
            token = {
                "text": data['text'][i],
                "left": int(data['left'][i]),
                "top": int(data['top'][i]),
                "width": int(data['width'][i]),
                "height": int(data['height'][i]),
                "confidence": float(data['conf'][i])
            }
            tokens.append(token)

    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(tokens, f, indent=2, ensure_ascii=False)

    return tokens


# -----------------------------
# Token Processing (Metadata + Tests)
# -----------------------------
def process_ocr_tokens(tokens):
    tokens = [t for t in tokens if all(k in t for k in ["top", "left", "text"])]

    if not tokens:
        return {"metadata": {}, "tests": []}

    # --- Step 1: group tokens into lines ---
    y_tolerance = 12
    tokens = sorted(tokens, key=lambda x: x["top"])
    lines = []
    current_line = []
    current_y = None
    for tok in tokens:
        if current_y is None:
            current_y = tok["top"]
            current_line.append(tok)
        elif abs(tok["top"] - current_y) <= y_tolerance:
            current_line.append(tok)
        else:
            lines.append(sorted(current_line, key=lambda t: t["left"]))
            current_line = [tok]
            current_y = tok["top"]
    if current_line:
        lines.append(sorted(current_line, key=lambda t: t["left"]))

    # --- Step 2: extract metadata ---
    def extract_metadata(lines, top_lines=20):
        data = {field: "NA" for field in ["name", "age", "gender", "patient_id", "lab_id", "mobile"]}
        table_keywords = ["test", "results", "unit", "ref", "interval", "parameter"]
        field_keywords = {
            "name": ["Name"],
            "age": ["Age"],
            "gender": ["Gender"],
            "patient_id": ["Pt ID", "PtID", "Patient ID", "Pt"],
            "lab_id": ["Lab Id", "Lab ID", "Lab No", "Lab No.", "LabNo", "labno", "LabID"],
            "mobile": ["Mob No", "Mobile"]
        }

        def extract_field_from_tokens(line_tokens, keywords):
            for i, tok in enumerate(line_tokens):
                for field, kws in keywords.items():
                    if any(kw.lower() in tok["text"].lower() for kw in kws):
                        value_tokens = []
                        for j in range(i + 1, len(line_tokens)):
                            t = line_tokens[j]["text"].strip()
                            if t not in [":", ".", "-", "–"]:
                                value_tokens.append(t)
                        if value_tokens:
                            return field, " ".join(value_tokens)
            return None, None

        for line in lines[:top_lines]:
            text_line = " ".join([t["text"] for t in line])
            if any(k.lower() in text_line.lower() for k in table_keywords):
                continue
            field, value = extract_field_from_tokens(line, field_keywords)
            if field and value:
                data[field] = value

        # --- Post-processing ---
        if data["lab_id"] != "NA":
            data["lab_id"] = re.sub(r"\D", "", data["lab_id"].replace("O", "0"))
        if data["patient_id"] != "NA":
            data["patient_id"] = re.sub(r"[^\w\d]", "", data["patient_id"])
        if data["age"] != "NA":
            data["age"] = data["age"].replace("O", "0")
        if data["gender"] != "NA":
            g = data["gender"].upper()
            if g in ["M", "MALE"]:
                data["gender"] = "Male"
            elif g in ["F", "FEMALE"]:
                data["gender"] = "Female"

        # Name: take first line with 'Name', combine with next lines if necessary
        name_lines = []
        for line in lines[:top_lines]:
            if any("name" in t["text"].lower() for t in line):
                name_lines.append(" ".join([t["text"] for t in line if t["text"].strip() not in [":"]]))
        if name_lines:
            data["name"] = " ".join(name_lines).replace("Name", "").strip()

        return data

    meta = extract_metadata(lines)

    # --- Step 3: extract tests ---
    def clean_unit_and_reference(unit_text, ref_text):
        unit_text, ref_text = unit_text.strip(), ref_text.strip()
        if len(unit_text.split()) > 2 or (re.search(r"[A-Za-z]{3,}", unit_text) and not re.match(r"^[A-Za-z]+\/[A-Za-z]+$", unit_text)):
            parts = unit_text.split(maxsplit=1)
            new_unit = parts[0]
            new_ref  = (ref_text + " " + parts[1]) if len(parts) > 1 else ref_text
            return new_unit.strip(), new_ref.strip()
        if re.search(r"\d", unit_text):
            parts = unit_text.split()
            unit_parts = [p for p in parts if not re.search(r"\d", p)]
            ref_parts  = [p for p in parts if re.search(r"\d", p)]
            new_unit = " ".join(unit_parts)
            new_ref  = (ref_text + " " + " ".join(ref_parts)).strip()
            return new_unit.strip() if new_unit else "NA", new_ref.strip() if new_ref else "NA"
        return unit_text if unit_text else "NA", ref_text if ref_text else "NA"

    def extract_tests(lines):
        tests = []
        candidate_lines = [line for line in lines if re.search(r"\d", " ".join(t["text"] for t in line))]
        if not candidate_lines:
            return tests

        x_positions = []
        for line in candidate_lines:
            for tok in line:
                x_positions.append([tok["left"]])
        x_positions = np.array(x_positions)

        kmeans = KMeans(n_clusters=4, random_state=0, n_init="auto").fit(x_positions)
        col_centers = sorted(kmeans.cluster_centers_.flatten())

        i = 0
        while i < len(candidate_lines):
            line = candidate_lines[i]
            cols = {j: [] for j in range(4)}
            for tok in line:
                col_idx = np.argmin([abs(tok["left"] - c) for c in col_centers])
                cols[col_idx].append(tok["text"])
            test_name = " ".join(cols[0]).strip()
            value     = " ".join(cols[1]).strip()
            unit      = " ".join(cols[2]).strip()
            reference = " ".join(cols[3]).strip()

            # merge multi-line references
            j = i + 1
            while j < len(candidate_lines):
                next_line = candidate_lines[j]
                next_cols = {k: [] for k in range(4)}
                for tok in next_line:
                    col_idx = np.argmin([abs(tok["left"] - c) for c in col_centers])
                    next_cols[col_idx].append(tok["text"])
                next_test = " ".join(next_cols[0]).strip()
                next_val  = " ".join(next_cols[1]).strip()
                if next_test or next_val:
                    break
                if next_cols[3]:
                    reference += " " + " ".join(next_cols[3])
                j += 1

            unit, reference = clean_unit_and_reference(unit, reference)

            if test_name and value:
                tests.append({
                    "test_name": test_name,
                    "value": value,
                    "unit": unit,
                    "reference": reference.strip()
                })
            i = j
        return tests

    tests = extract_tests(lines)

    # --- Step 4: move metadata fields from tests if test_name matches ---
    field_keywords = {
        "name": ["Name"],
        "age": ["Age"],
        "gender": ["Gender"],
        "patient_id": ["Pt ID", "PtID", "Patient ID", "Pt"],
        "lab_id": ["Lab Id", "Lab ID", "Lab No", "Lab No.", "LabNo", "labno", "LabID"],
        "mobile": ["Mob No", "Mobile"]
    }

    remaining_tests = []
    for t in tests:
        test_field = t["test_name"].strip()
        moved = False
        for field, keywords in field_keywords.items():
            if any(kw.lower() in test_field.lower() for kw in keywords):
                if meta[field] == "NA":
                    meta[field] = t["value"].strip()
                moved = True
                break
        if not moved:
            remaining_tests.append(t)

    tests = remaining_tests

    return {"metadata": meta, "tests": tests}


# -----------------------------
# Main Pipeline
# -----------------------------
def run_pipeline_with_confidence(img_path, ner_model_path):
    """
    Full pipeline:
    1. Preprocess → OCR → Token Processing
    2. Add NER model predictions
    3. Return metadata, tests, and confidence scores
    """
    # Preprocess
    preprocessed = preprocess_image(img_path)

    # Save temp file for OCR
    tmp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    cv2.imwrite(tmp_img.name, preprocessed)

    # OCR
    tokens = ocr_image(tmp_img.name)
    os.remove(tmp_img.name)

    if not tokens:
        return {"metadata": {}, "tests": [], "confidence": {}}

    # Rule-based extraction
    rule_result = process_ocr_tokens(tokens)
    metadata, tests = rule_result["metadata"], rule_result["tests"]

    # NER model
    nlp = spacy.load(ner_model_path)
    text = " ".join([t["text"] for t in tokens])
    doc = nlp(text)

    confidence = {field: 0.0 for field in metadata.keys()}
    for ent in doc.ents:
        field = ent.label_.lower()
        value = ent.text.strip()
        if field in metadata and (metadata[field] in ["NA", ""]):
            metadata[field] = value
            confidence[field] = 0.9  # NER prediction

    # Fallback confidence
    for field, val in metadata.items():
        if val not in ["NA", ""] and confidence[field] == 0.0:
            confidence[field] = 0.5  # extracted by rules

    return {"metadata": metadata, "tests": tests, "confidence": confidence}
