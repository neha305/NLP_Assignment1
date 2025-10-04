import logo from './logo.svg';
import './App.css';
import React, { useState } from "react";

function App() {
  const [file, setFile] = useState(null);
  const [reportId, setReportId] = useState(null);
  const [jsonResult, setJsonResult] = useState(null);
  const [confirmJson, setConfirmJson] = useState("");

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const uploadFile = async () => {
    if (!file) {
      alert("Please select a PDF file first");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("http://localhost:8000/upload", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      setReportId(data.report_id);
      setJsonResult(data);
      setConfirmJson(JSON.stringify(data, null, 2));
    } catch (err) {
      console.error("Upload error:", err);
      alert("Upload failed");
    }
  };

  const confirmReport = async () => {
    if (!reportId) {
      alert("No report to confirm");
      return;
    }

    const formData = new FormData();
    formData.append("report_id", reportId);
    formData.append("corrected_json", confirmJson);

    try {
      const res = await fetch("http://localhost:8000/confirm", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      alert("Saved: " + JSON.stringify(data));
    } catch (err) {
      console.error("Confirm error:", err);
      alert("Confirm failed");
    }
  };

  return (
    <div className="container">
      <h2>Medical Report OCR Extraction</h2>

      {/* Custom File Input */}
      <input
        type="file"
        id="file-upload"
        className="file-input"
        onChange={handleFileChange}
      />
      <label htmlFor="file-upload" className="file-label">
        Choose PDF File
      </label>

      {/* Show selected file name */}
      {file && (
        <span className="selected-file">Selected File: {file.name}</span>
      )}

      {/* Upload button */}
      <button onClick={uploadFile} className="upload-button">
        Upload
      </button>

      {/* Show results, editable JSON, confirm button */}
      {jsonResult && (
        <div style={{ marginTop: "20px" }}>
          <h3 style={{ color: "#4b6cb7" }}>Extracted JSON</h3>
          <pre className="output-area">
            {JSON.stringify(jsonResult, null, 2)}
          </pre>
          <h3 style={{ color: "#28a745" }}>Edit & Confirm</h3>
          <textarea
            style={{ width: "100%", height: "250px" }}
            value={confirmJson}
            onChange={e => setConfirmJson(e.target.value)}
          />
          <br />
          <button
            onClick={confirmReport}
            className="confirm-button"
          >
            Confirm
          </button>
        </div>
      )}
    </div>
  );
}

export default App;
