from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

from fastapi.responses import JSONResponse, HTMLResponse
import shutil, os, uuid, json, tempfile
from pdf2image import convert_from_bytes
from pipeline import run_pipeline_with_confidence, ocr_image
import cv2

app = FastAPI(title="OCR Report Extractor", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

NER_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "spacy_model")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(os.path.join(RESULTS_DIR, "corrections"), exist_ok=True)

# ------------------------
# Upload & Extract
# ------------------------
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    with open(temp_pdf.name, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Convert PDF to images
    pages = convert_from_bytes(open(temp_pdf.name, "rb").read(), dpi=200)
    os.remove(temp_pdf.name)

    report_id = str(uuid.uuid4())
    all_results = []
    all_tokens = []

    for i, page in enumerate(pages):
        tmp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        page.save(tmp_img.name, "PNG")

        # Run pipeline
        result = run_pipeline_with_confidence(tmp_img.name, NER_MODEL_PATH)
        all_results.append(result)

        # Keep OCR tokens for later corrections
        tokens = ocr_image(tmp_img.name)
        all_tokens.append(tokens)

        os.remove(tmp_img.name)

    response = {
        "report_id": report_id,
        "pages": all_results
    }

    # Save raw JSON + tokens for later correction
    pending_file = os.path.join(RESULTS_DIR, f"pending_{report_id}.json")
    with open(pending_file, "w") as f:
        json.dump({"report": response, "tokens": all_tokens}, f, indent=2)

    return JSONResponse(content=response)


# ------------------------
# Confirm & Save Corrections
# ------------------------
@app.post("/confirm")
async def confirm_report(report_id: str = Form(...), corrected_json: str = Form(...)):
    corrected = json.loads(corrected_json)

    # Save confirmed JSON
    confirmed_path = os.path.join(RESULTS_DIR, f"confirmed_{report_id}.json")
    with open(confirmed_path, "w") as f:
        json.dump(corrected, f, indent=2)

    # Also save corrections (original tokens + initial extracted + corrected)
    pending_file = os.path.join(RESULTS_DIR, f"pending_{report_id}.json")
    if os.path.exists(pending_file):
        with open(pending_file, "r") as f:
            pending_data = json.load(f)

        correction_obj = {
            "original_tokens": pending_data["tokens"],
            "extracted": pending_data["report"],
            "corrected": corrected
        }

        correction_path = os.path.join(RESULTS_DIR, "corrections", f"correction_{report_id}.json")
        with open(correction_path, "w") as f:
            json.dump(correction_obj, f, indent=2)

    return {"status": "saved", "confirmed": confirmed_path}
