# app.py
import os
import re
import io
import sqlite3
from typing import List
from flask import Flask, request, jsonify, render_template, redirect, url_for
import pdfplumber
import docx
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
import yake
import spacy
from transformers import pipeline

# ---------------- CONFIG ----------------
UPLOAD_FOLDER = "./uploads"
DB_PATH = "resumes.db"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

KEYWORD_COVERAGE_WEIGHT = 0.5
FORMAT_WEIGHT = 0.2
READABILITY_WEIGHT = 0.15
EXPERIENCE_MATCH_WEIGHT = 0.15

# ---------------- MODELS ----------------
# These will download on first run if not present
embedder = SentenceTransformer("all-MiniLM-L6-v2")
nlp = spacy.load("en_core_web_sm")
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
kw_extractor = yake.KeywordExtractor(lan="en", n=1, top=50)

# ---------------- APP INIT ----------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ---------------- DB ----------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS resumes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT,
        text_snippet TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()
    conn.close()

init_db()

# ---------------- Helpers ----------------
def extract_text_from_pdf(path_or_bytes) -> str:
    text = []
    if isinstance(path_or_bytes, (bytes, io.BytesIO)):
        f = io.BytesIO(path_or_bytes if isinstance(path_or_bytes, bytes) else path_or_bytes.read())
        with pdfplumber.open(f) as pdf:
            for p in pdf.pages:
                t = p.extract_text()
                if t:
                    text.append(t)
    else:
        with pdfplumber.open(path_or_bytes) as pdf:
            for p in pdf.pages:
                t = p.extract_text()
                if t:
                    text.append(t)
    return "\n".join(text)

def extract_text_from_docx(path) -> str:
    doc = docx.Document(path)
    paragraphs = [p.text for p in doc.paragraphs if p.text]
    return "\n".join(paragraphs)

def extract_text_from_file(file_storage) -> str:
    filename = file_storage.filename.lower()
    data = file_storage.read()
    file_storage.stream.seek(0)
    try:
        if filename.endswith(".pdf"):
            return extract_text_from_pdf(data)
        elif filename.endswith(".docx"):
            tmp = os.path.join(UPLOAD_FOLDER, filename)
            file_storage.save(tmp)
            return extract_text_from_docx(tmp)
        elif filename.endswith(".txt"):
            return data.decode(errors='ignore')
        else:
            # try pdf first
            try:
                return extract_text_from_pdf(data)
            except Exception:
                return data.decode(errors='ignore')
    except Exception:
        return ""

def save_resume_to_db(filename, text_snippet):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO resumes (filename, text_snippet) VALUES (?, ?)", (filename, text_snippet[:1000]))
    conn.commit()
    conn.close()

def basic_format_checks(text: str) -> dict:
    issues = []
    score = 100
    if not re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text):
        issues.append("No email address found.")
        score -= 15
    if not re.search(r"\b\d{7,15}\b", text):
        issues.append("No phone number found.")
        score -= 10
    words = len(text.split())
    if words < 150:
        issues.append("Resume is short (<150 words). Add more detail.")
        score -= 10
    if words < 50:
        issues.append("Very little selectable text; resume may be a scanned image.")
        score -= 20
    if words > 3000:
        issues.append("Resume appears very long (>3000 words). Consider concise formatting.")
        score -= 10
    return {"issues": issues, "format_score": max(0, score)}

def extract_keywords(text: str, top_k: int = 30) -> List[str]:
    try:
        kws = kw_extractor.extract_keywords(text)
        top = [k for k, _ in kws][:top_k]
        return top
    except Exception:
        # fallback to simple split
        words = re.findall(r"\b[A-Za-z][A-Za-z0-9+#\.\-]{1,}\b", text.lower())
        freq = {}
        for w in words:
            freq[w] = freq.get(w, 0) + 1
        sorted_k = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [w for w, _ in sorted_k[:top_k]]

def extract_skills(text: str, top_k: int = 80) -> List[str]:
    doc = nlp(text)
    candidates = set()
    for nc in doc.noun_chunks:
        tok = nc.text.strip().lower()
        if len(tok.split()) <= 4 and len(tok) > 2:
            candidates.add(re.sub(r'\s+', ' ', tok))
    for k in extract_keywords(text, top_k=top_k):
        candidates.add(k.lower())
    stop_phrases = {"experience", "years", "work", "team", "skills", "knowledge"}
    filtered = [c for c in candidates if c not in stop_phrases]
    return filtered[:top_k]

def extract_experience_years(text: str) -> int:
    m = re.search(r"(\d{1,2})\s*\+?\s*years", text.lower())
    if m:
        return int(m.group(1))
    return 0

def compute_ats_score(resume_text: str, jd_text: str = None) -> dict:
    if jd_text:
        jd_kws = extract_keywords(jd_text, top_k=60)
    else:
        jd_kws = extract_keywords(resume_text, top_k=40)
    resume_kws = [k.lower() for k in extract_keywords(resume_text, top_k=200)]
    matched = [k for k in jd_kws if any(k in r for r in resume_kws)]
    coverage = len(matched) / max(1, len(jd_kws))
    keyword_score = int(coverage * 100)

    fmt = basic_format_checks(resume_text)
    fmt_score = fmt['format_score']

    sentences = re.split(r'[.!?\n]+', resume_text)
    avg_sent_len = sum(len(s.split()) for s in sentences if s.strip()) / max(1, len([s for s in sentences if s.strip()]))
    if avg_sent_len < 10:
        read_score = 80
    elif avg_sent_len <= 22:
        read_score = 100
    else:
        read_score = max(30, int(140 - avg_sent_len * 3))

    exp_needed = 0
    if jd_text:
        m = re.search(r"(\d{1,2})\s*\+?\s*years", jd_text.lower())
        if m:
            exp_needed = int(m.group(1))
    resume_exp = extract_experience_years(resume_text)
    if exp_needed > 0:
        if resume_exp >= exp_needed:
            exp_score = 100
        else:
            exp_score = int(max(0, 100 * (resume_exp / exp_needed)))
    else:
        exp_score = 80

    final = (keyword_score * KEYWORD_COVERAGE_WEIGHT +
             fmt_score * FORMAT_WEIGHT +
             read_score * READABILITY_WEIGHT +
             exp_score * EXPERIENCE_MATCH_WEIGHT)
    final_int = int(max(0, min(100, final)))

    details = {
        "final_score": final_int,
        "keyword_score": int(keyword_score),
        "format_score": int(fmt_score),
        "readability_score": int(read_score),
        "experience_score": int(exp_score),
        "matched_keywords": matched,
        "num_jd_keywords": len(jd_kws),
        "num_matched": len(matched),
        "format_issues": fmt['issues'],
        "resume_experience_years": resume_exp,
        "jd_experience_years": exp_needed
    }
    return details

def compute_semantic_match(resume_text: str, jd_text: str) -> dict:
    resume_chunks = [p for p in re.split(r'\n{1,}', resume_text) if len(p.split()) > 6]
    jd_chunks = [p for p in re.split(r'\n{1,}', jd_text) if len(p.split()) > 6]
    if not resume_chunks:
        resume_chunks = [resume_text]
    if not jd_chunks:
        jd_chunks = [jd_text]
    emb_resume = embedder.encode(resume_chunks, convert_to_tensor=True)
    emb_jd = embedder.encode(jd_chunks, convert_to_tensor=True)
    sim_matrix = util.cos_sim(emb_resume, emb_jd).cpu().numpy()
    import numpy as np
    best = float(sim_matrix.max()) if sim_matrix.size else 0.0
    avg_per_jd = float(np.max(sim_matrix, axis=0).mean()) if sim_matrix.size else 0.0
    best_score = int(best * 100)
    avg_score = int(avg_per_jd * 100)
    return {"best_similarity": best_score, "avg_similarity": avg_score}

def generate_suggestions(resume_text: str, jd_text: str = None, matched_keywords: List[str] = None) -> dict:
    try:
        summary = summarizer(resume_text[:1024], max_length=60, min_length=20, do_sample=False)[0]['summary_text']
    except Exception:
        summary = " ".join(resume_text.splitlines()[:4])
    suggestions = []
    if matched_keywords:
        if len(matched_keywords) < 5 and matched_keywords:
            suggestions.append("Add more role-specific keywords from the JD such as: " + ", ".join(matched_keywords[:10]))
    fmt = basic_format_checks(resume_text)
    for issue in fmt["issues"]:
        suggestions.append(issue)
    suggestions += [
        "Use exact keywords from the JD within context rather than in a skills blob.",
        "Prefer bullet points with achievements and quantified outcomes (numbers, %, counts).",
        "Ensure contact info is at top: name, email, phone, location (city, country).",
        "Use standard section headings: Summary, Experience, Education, Skills, Certifications."
    ]
    return {"summary": summary, "suggestions": suggestions[:12]}

def search_naukri(city: str, query: str = None, max_results: int = 10) -> List[dict]:
    results = []
    base = "https://www.naukri.com/{query}-jobs-in-{city}".format(query=(query or "jobs").replace(" ", "-"), city=city.replace(" ", "-"))
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(base, headers=headers, timeout=10)
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, "html.parser")
            cards = soup.select(".jobTuple")[:max_results] or soup.select(".info")[:max_results]
            for c in cards[:max_results]:
                title = c.select_one(".title") or c.select_one("a")
                title_text = title.get_text(strip=True) if title else ""
                link = title.get("href") if title and title.has_attr("href") else (c.select_one("a").get("href") if c.select_one("a") else "")
                company = c.select_one(".company") or c.select_one(".companyInfo")
                company_text = company.get_text(strip=True) if company else ""
                loc = c.select_one(".location") or c.select_one(".loc")
                loc_text = loc.get_text(strip=True) if loc else city
                results.append({"title": title_text, "company": company_text, "location": loc_text, "link": link})
    except Exception as e:
        results.append({"error": f"Failed to fetch Naukri results: {str(e)}"})
    return results

def search_linkedin_like(city: str, query: str = None, max_results: int = 10) -> List[dict]:
    results = []
    try:
        search_url = "https://www.google.com/search?q=site:linkedin.com/jobs+{query}+in+{city}".format(
            query=(query or "jobs").replace(" ", "+"), city=city.replace(" ", "+")
        )
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(search_url, headers=headers, timeout=8)
        soup = BeautifulSoup(r.text, "html.parser")
        anchors = soup.select("a")
        found = 0
        for a in anchors:
            href = a.get("href") or ""
            if href.startswith("/url?q="):
                real = href.split("/url?q=")[1].split("&sa=")[0]
                if "linkedin.com/jobs" in real:
                    title = a.get_text(strip=True) or "LinkedIn Job"
                    results.append({"title": title, "link": real})
                    found += 1
                    if found >= max_results:
                        break
    except Exception as e:
        results.append({"error": f"LinkedIn-style search failed: {str(e)}"})
    return results

# ---------------- Routes (UI + API) ----------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ats", methods=["GET", "POST"])
def ats_page():
    if request.method == "POST":
        if 'file' not in request.files:
            return redirect(url_for('ats_page'))
        f = request.files['file']
        text = extract_text_from_file(f)
        path = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
        f.stream.seek(0)
        f.save(path)
        save_resume_to_db(f.filename, text[:1000])
        ats = compute_ats_score(text)
        suggestions = generate_suggestions(text, matched_keywords=ats.get("matched_keywords", []))
        return render_template("result_ats.html", ats=ats, suggestions=suggestions, resume_text=text[:3000])
    return render_template("ats.html")

@app.route("/jd", methods=["GET", "POST"])
def jd_page():
    if request.method == "POST":
        if 'file' not in request.files:
            return redirect(url_for('jd_page'))
        f = request.files['file']
        resume_text = extract_text_from_file(f)
        jd_text = ""
        if request.form.get("jd_text"):
            jd_text = request.form.get("jd_text")
        elif request.form.get("jd_url"):
            jd_url = request.form.get("jd_url")
            try:
                r = requests.get(jd_url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
                soup = BeautifulSoup(r.text, "html.parser")
                for s in soup(["script", "style", "noscript"]):
                    s.decompose()
                jd_text = " ".join([p.get_text(" ", strip=True) for p in soup.find_all(["p", "li", "div"])])
            except Exception:
                jd_text = ""
        if not jd_text:
            return render_template("jd.html", error="No JD text found. Provide JD text or JD URL.")
        path = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
        f.stream.seek(0)
        f.save(path)
        save_resume_to_db(f.filename, resume_text[:1000])
        ats_details = compute_ats_score(resume_text, jd_text)
        sem = compute_semantic_match(resume_text, jd_text)
        suggestions = generate_suggestions(resume_text, jd_text, matched_keywords=ats_details.get("matched_keywords"))
        return render_template("result_jd.html", ats=ats_details, sem=sem, suggestions=suggestions, jd_text=jd_text[:3000], resume_text=resume_text[:3000])
    return render_template("jd.html")

@app.route("/jobsearch", methods=["GET", "POST"])
def jobsearch_page():
    if request.method == "POST":
        city = request.form.get("city")
        if not city:
            return render_template("jobsearch.html", error="Please enter a city.")
        resume_text = ""
        if 'file' in request.files and request.files['file'].filename != "":
            f = request.files['file']
            resume_text = extract_text_from_file(f)
            path = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
            f.stream.seek(0)
            f.save(path)
            save_resume_to_db(f.filename, resume_text[:1000])
        query = request.form.get("query") or None
        if not query and resume_text:
            skills = extract_skills(resume_text, top_k=8)
            query = " ".join(skills[:3]) if skills else "software engineer"
        naukri = search_naukri(city, query=query, max_results=10)
        linkedin_like = search_linkedin_like(city, query=query, max_results=8)
        return render_template("result_job.html", query=query, naukri=naukri, linkedin_like=linkedin_like, city=city)
    return render_template("jobsearch.html")

# Minimal API endpoints (optional)
@app.route("/api/ats-check", methods=["POST"])
def api_ats_check():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    f = request.files['file']
    text = extract_text_from_file(f)
    path = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
    f.stream.seek(0)
    f.save(path)
    save_resume_to_db(f.filename, text[:1000])
    ats = compute_ats_score(text)
    suggestions = generate_suggestions(text, matched_keywords=ats.get("matched_keywords", []))
    return jsonify({"ats": ats, "suggestions": suggestions})

@app.route("/api/jd-match", methods=["POST"])
def api_jd_match():
    if 'file' not in request.files:
        return jsonify({"error": "No resume file provided"}), 400
    f = request.files['file']
    resume_text = extract_text_from_file(f)
    jd_text = ""
    if request.form.get("jd_text"):
        jd_text = request.form.get("jd_text")
    elif request.form.get("jd_url"):
        jd_url = request.form.get("jd_url")
        try:
            r = requests.get(jd_url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
            soup = BeautifulSoup(r.text, "html.parser")
            for s in soup(["script", "style", "noscript"]):
                s.decompose()
            jd_text = " ".join([p.get_text(" ", strip=True) for p in soup.find_all(["p", "li", "div"])])
        except Exception:
            jd_text = ""
    if not jd_text:
        return jsonify({"error": "No JD text found. Provide jd_text or jd_url."}), 400
    path = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
    f.stream.seek(0)
    f.save(path)
    save_resume_to_db(f.filename, resume_text[:1000])
    ats_details = compute_ats_score(resume_text, jd_text)
    sem = compute_semantic_match(resume_text, jd_text)
    suggestions = generate_suggestions(resume_text, jd_text, matched_keywords=ats_details.get("matched_keywords"))
    return jsonify({"ats_details": ats_details, "semantic_match": sem, "suggestions": suggestions})

@app.route("/api/job-search", methods=["POST"])
def api_job_search():
    city = request.form.get("city") or request.args.get("city")
    if not city:
        return jsonify({"error": "Provide city parameter (e.g., city=Pune)"}), 400
    resume_text = ""
    if 'file' in request.files:
        f = request.files['file']
        resume_text = extract_text_from_file(f)
        path = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
        f.stream.seek(0)
        f.save(path)
        save_resume_to_db(f.filename, resume_text[:1000])
    query = request.form.get("query") or None
    if not query and resume_text:
        skills = extract_skills(resume_text, top_k=8)
        query = " ".join(skills[:3]) if skills else "software engineer"
    naukri = search_naukri(city, query=query, max_results=10)
    linkedin_like = search_linkedin_like(city, query=query, max_results=8)
    return jsonify({"query_used": query, "naukri_results": naukri, "linkedin_like_results": linkedin_like})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
