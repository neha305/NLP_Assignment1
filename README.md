Medical Report OCR Extraction for NLP Home Assignment

Steps to run the project:

```bash
# Clone the repository
git clone https://github.com/neha305/NLP_Assignment1

# Move into the directory
cd NLP_Assignment1

# Install dependencies (might have to install some packages like Node/npm separately)
pip install -r requirements.txt

# Move into the backend directory
cd backend

# Run the backend
uvicorn main:app --reload --port 8000

# Run the frontend from the project directory from a different terminal
npm start

# Webpage should automatically open, if not, open the following in a browser
http://localhost:3000

You can now upload a PDF, see the extracted JSONs and review them, corrections will be placed in the results folder.
