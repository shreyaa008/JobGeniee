import os
import shutil
import traceback

import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Student', 'Backend'))

# ── Import Student modules ────────────────────────────────────────────────────
from predictor import predict_student

# ── Import Recruiter modules ──────────────────────────────────────────────────
from config import (
    NER_MODEL_PATH,
    RANKING_MODEL_PATH,
    FEATURE_COLS_PATH,
    SCALER_PATH,
    UPLOADS_DIR,
)
from utils.parser import load_ner_model, parse_resume
from utils.ranker import rank_uploaded_resumes

app = Flask(__name__)
CORS(app)

# ── Load recruiter models once at startup ─────────────────────────────────────
nlp, use_ner  = load_ner_model(NER_MODEL_PATH)
rec_model     = joblib.load(RANKING_MODEL_PATH)
rec_scaler    = joblib.load(SCALER_PATH)
feature_cols  = joblib.load(FEATURE_COLS_PATH)
print("All models loaded ✓")

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/')
def landing():
    """Role selection landing page"""
    return render_template('landing.html')

@app.route('/login')
def login():
    """Firebase login/signup page"""
    return render_template('login.html')

@app.route('/student')
def student():
    """Student predictor form"""
    return render_template('student_index.html')

@app.route('/student/result')
def student_result():
    """Student result dashboard"""
    return render_template('student_result.html')

@app.route('/recruiter')
def recruiter():
    """Recruiter resume screener form"""
    return render_template('index.html')

@app.route('/recruiter/results')
def recruiter_results_page():
    """Recruiter results page (rendered via POST redirect or JS)"""
    return render_template('results.html')



@app.route('/health')
def health():
    return jsonify({"status": "running"})

# ── Student API ───────────────────────────────────────────────────────────────

@app.route('/api/student/predict', methods=['POST'])
def student_predict():
    try:
        data = request.get_json()
        required = ['college_tier', 'cgpa', 'internships', 'github_projects',
                    'backlogs', 'hackathons', 'certifications',
                    'aptitude_score', 'skills']
        missing = [f for f in required if f not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        if data['college_tier'] not in ['Tier 1', 'Tier 2', 'Tier 3']:
            return jsonify({"error": "college_tier must be Tier 1, Tier 2, or Tier 3"}), 400
        if not (4.0 <= float(data['cgpa']) <= 10.0):
            return jsonify({"error": "CGPA must be between 4.0 and 10.0"}), 400
        if not isinstance(data['skills'], list) or len(data['skills']) == 0:
            return jsonify({"error": "Select at least one skill"}), 400

        result = predict_student(
            college_tier    = data['college_tier'],
            cgpa            = float(data['cgpa']),
            internships     = int(data['internships']),
            github_projects = int(data['github_projects']),
            backlogs        = int(data['backlogs']),
            hackathons      = int(data['hackathons']),
            certifications  = int(data['certifications']),
            aptitude_score  = float(data['aptitude_score']),
            skills          = data['skills']
        )
        return jsonify(result), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ── Recruiter API ─────────────────────────────────────────────────────────────

@app.route('/api/recruiter/screen', methods=['POST'])
def recruiter_screen():
    try:
        min_exp    = float(request.form.get("min_exp", 0) or 0)
        min_salary = float(request.form.get("min_salary", 0) or 0)
        max_salary = float(request.form.get("max_salary", 10_000_000) or 10_000_000)
        top_n      = max(1, min(int(request.form.get("top_n", 5) or 5), 50))

        if min_salary > max_salary:
            min_salary, max_salary = max_salary, min_salary

        skills_raw      = request.form.get("skills", "")
        required_skills = [s.strip() for s in skills_raw.split(",") if s.strip()]

        os.makedirs(UPLOADS_DIR, exist_ok=True)
        uploaded_files = request.files.getlist("resumes")
        parsed_list    = []

        for pdf_file in uploaded_files:
            if pdf_file and pdf_file.filename.endswith(".pdf"):
                save_path = os.path.join(UPLOADS_DIR, pdf_file.filename)
                try:
                    pdf_file.save(save_path)
                    parsed = parse_resume(save_path, nlp, use_ner)
                    if parsed:
                        parsed_list.append(parsed)
                except Exception as e:
                    print(f"Could not process {pdf_file.filename}: {e}")

        try:
            shutil.rmtree(UPLOADS_DIR)
        except Exception:
            pass

        if not required_skills and parsed_list:
            all_skills = []
            for p in parsed_list:
                all_skills.extend(p.get("skills", []))
            required_skills = list(dict.fromkeys(all_skills))

        ranked = []
        if parsed_list:
            all_ranked = rank_uploaded_resumes(
                parsed_list, required_skills, rec_model, rec_scaler, feature_cols
            )
            ranked = [r for r in all_ranked if r["experience"] >= min_exp][:top_n]
            for i, r in enumerate(ranked, 1):
                r["rank"] = i

        return render_template('results.html',
    ranked=ranked,
    required_skills=required_skills,
    min_exp=min_exp,
    min_salary=int(min_salary),
    max_salary=int(max_salary),
    top_n=top_n,
    total_uploaded=len(parsed_list),
), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    print("JobGenie unified → http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)