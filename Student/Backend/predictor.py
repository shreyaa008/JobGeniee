import pickle
import json
import os
import numpy as np
import pandas as pd
from improvements import get_improvements

# ── Paths — update MODELS_DIR if needed ──────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, '..', 'Models')

# ── Load models once at startup (not on every request) ───────────────────────
def _load_models():
    sal_path  = os.path.join(MODELS_DIR, 'salary_model.pkl')
    role_path = os.path.join(MODELS_DIR, 'role_model.pkl')
    le_path   = os.path.join(MODELS_DIR, 'label_encoder_role.pkl')
    meta_path = os.path.join(MODELS_DIR, 'feature_columns.json')

    for path in [sal_path, role_path, le_path, meta_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model file not found: {path}\n"
                f"Run 04_model_training_2.ipynb first."
            )

    with open(sal_path,  'rb') as f: sal_model  = pickle.load(f)
    with open(role_path, 'rb') as f: role_model = pickle.load(f)
    with open(le_path,   'rb') as f: le         = pickle.load(f)
    with open(meta_path, 'r')  as f: meta       = json.load(f)

    print("✅ Models loaded successfully")
    print(f"   Salary model : {meta['salary_model']}  (R²={meta['salary_r2']})")
    print(f"   Role model   : {meta['role_model']}  (Acc={meta['role_accuracy']*100:.1f}%)")

    return sal_model, role_model, le, meta

# Load once when Flask starts
_sal_model, _role_model, _le, _meta = _load_models()

SALARY_FEATURES = _meta['salary_features']
ROLE_FEATURES   = _meta['role_features']
ALL_SKILLS      = _meta['all_skills']
TIER_MAPPING    = _meta['tier_mapping']

ADVANCED     = {'DSA', 'Machine Learning', 'Deep Learning', 'TensorFlow', 'PyTorch'}
INTERMEDIATE = {'Python', 'Java', 'React', 'AWS', 'Docker', 'Node.js', 'Django'}
BASIC        = set(ALL_SKILLS) - ADVANCED - INTERMEDIATE


# ── Build salary input row ────────────────────────────────────────────────────
def _build_salary_row(college_tier, cgpa, internships, github_projects,
                      backlogs, hackathons, certifications,
                      aptitude_score, skills):
    skill_set = set(skills)
    row = {
        'college_tier_encoded': TIER_MAPPING[college_tier],
        'cgpa':                 cgpa,
        'internships':          internships,
        'github_projects':      github_projects,
        'backlogs':             backlogs,
        'hackathons':           hackathons,
        'certifications':       certifications,
        'advanced_skills':      len(skill_set & ADVANCED),
        'intermediate_skills':  len(skill_set & INTERMEDIATE),
        'basic_skills':         len(skill_set & BASIC),
        'aptitude_score':       aptitude_score,
    }
    return pd.DataFrame([row])[SALARY_FEATURES]


# ── Build role input row ──────────────────────────────────────────────────────
def _build_role_row(college_tier, cgpa, internships, github_projects,
                    backlogs, hackathons, certifications,
                    aptitude_score, skills):
    skill_set = set(skills)
    row = {
        'college_tier_encoded': TIER_MAPPING[college_tier],
        'cgpa':                 cgpa,
        'internships':          internships,
        'github_projects':      github_projects,
        'backlogs':             backlogs,
        'hackathons':           hackathons,
        'certifications':       certifications,
        'aptitude_score':       aptitude_score,
    }
    for skill in ALL_SKILLS:
        col      = 'skill_' + skill.lower().replace('/', '_').replace('.', '_').replace(' ', '_')
        row[col] = 1 if skill in skill_set else 0

    return pd.DataFrame([row])[ROLE_FEATURES]


# ── Main prediction function ──────────────────────────────────────────────────
def predict_student(college_tier, cgpa, internships, github_projects,
                    backlogs, hackathons, certifications,
                    aptitude_score, skills):
    """
    Runs salary + role prediction and returns a clean dict for the API.
    """
    sal_df  = _build_salary_row(college_tier, cgpa, internships, github_projects,
                                 backlogs, hackathons, certifications,
                                 aptitude_score, skills)
    role_df = _build_role_row(college_tier, cgpa, internships, github_projects,
                               backlogs, hackathons, certifications,
                               aptitude_score, skills)

    # ── Salary prediction ─────────────────────────────────────────────────────
    predicted_salary = round(float(_sal_model.predict(sal_df)[0]), 2)

    # ── Role prediction + top-3 with confidence ───────────────────────────────
    role_encoded   = _role_model.predict(role_df)[0]
    predicted_role = _le.inverse_transform([role_encoded])[0]

    role_probs = _role_model.predict_proba(role_df)[0]
    top3_idx   = np.argsort(role_probs)[::-1][:3]
    top_roles  = [
        {
            "role":       _le.classes_[i],
            "confidence": round(float(role_probs[i]) * 100, 1)
        }
        for i in top3_idx
    ]

    # ── Improvement suggestions ───────────────────────────────────────────────
    improvements = get_improvements(
        cgpa=cgpa, skills=skills, internships=internships,
        github_projects=github_projects, hackathons=hackathons,
        backlogs=backlogs, certifications=certifications,
        aptitude_score=aptitude_score
    )

    return {
        "predicted_salary_lpa": predicted_salary,
        "predicted_role":       predicted_role,
        "top_roles":            top_roles,
        "improvements":         improvements
    }