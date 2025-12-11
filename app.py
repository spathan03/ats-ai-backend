#!/usr/bin/env python3
from flask import Flask, request, jsonify
from flask_cors import CORS
import pdfplumber
import docx
import os
import PyPDF2
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = SentenceTransformer('all-MiniLM-L6-v2')


# --------------------------
# Helper Functions
# --------------------------

def read_pdf(file_path):
    try:
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text
    except:
        text = ""
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
        return text


def read_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join([p.text for p in doc.paragraphs])


def read_text(file_path):
    with open(file_path, "r", errors="ignore") as f:
        return f.read()


def extract_text(file):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    if file.filename.endswith(".pdf"):
        return read_pdf(file_path)
    elif file.filename.endswith(".docx"):
        return read_docx(file_path)
    elif file.filename.endswith(".txt"):
        return read_text(file_path)
    else:
        return ""


# --------------------------
# 📌 API 1 — ATS SCORE
# --------------------------
@app.route("/ats-score", methods=["POST"])
def ats_score():
    resume_file = request.files.get("resume")
    jd_file = request.files.get("jd")

    if not resume_file or not jd_file:
        return jsonify({"error": "Files missing"}), 400

    resume_text = extract_text(resume_file)
    jd_text = extract_text(jd_file)

    emb1 = model.encode(resume_text, convert_to_tensor=True)
    emb2 = model.encode(jd_text, convert_to_tensor=True)

    score = float(util.pytorch_cos_sim(emb1, emb2).item() * 100)

    return jsonify({
        "match_score": round(score, 2),
        "resume_text": resume_text[:5000],
        "jd_text": jd_text[:5000]
    })


# --------------------------
# 📌 API 2 — JD → Resume Match Score (Separate)
# --------------------------
@app.route("/jd-match", methods=["POST"])
def jd_match():
    resume_file = request.files.get("resume")
    jd_text = request.form.get("jd")

    resume_text = extract_text(resume_file)

    emb1 = model.encode(resume_text, convert_to_tensor=True)
    emb2 = model.encode(jd_text, convert_to_tensor=True)

    score = float(util.pytorch_cos_sim(emb1, emb2).item() * 100)

    return jsonify({
        "match_score": round(score, 2)
    })


# --------------------------
# 📌 API 3 — Job Search (Naukri + LinkedIn-Like)
# --------------------------
@app.route("/job-search", methods=["POST"])
def job_search():
    data = request.json
    keywords = data.get("keywords", "")
    location = data.get("location", "")

    # Dummy data (You can connect real APIs later)
    naukri_jobs = [
        {"title": "Automation Analyst", "company": "TechCorp", "location": location},
        {"title": "Workflow Engineer", "company": "InfyTech", "location": location},
    ]

    linkedin_jobs = [
        {"title": "Senior Workflow Specialist", "company": "DataFlow Ltd", "location": location},
        {"title": "Process Automation Expert", "company": "CloudBridge", "location": location},
    ]

    return jsonify({
        "keywords_used": keywords,
        "naukri": naukri_jobs,
        "linkedin": linkedin_jobs
    })


# --------------------------
# Render Run Command
# --------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
