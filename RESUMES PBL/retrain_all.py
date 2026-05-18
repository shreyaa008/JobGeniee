"""
retrain_all.py — Run this ANY TIME you change Python or scikit-learn version.
Place this file in: RESUMES PBL/
Run with:  python retrain_all.py
"""

import os, sys, json, pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score, accuracy_score
from xgboost import XGBClassifier
import joblib

BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
STUDENT_DATA     = os.path.join(BASE_DIR, '..', 'Student', 'Dataset', 'raw', 'student_salary_dataset.csv')
STUDENT_MODELS   = os.path.join(BASE_DIR, '..', 'Student', 'Models')
RECRUITER_DATA   = os.path.join(BASE_DIR, 'data', 'candidates.csv')
RECRUITER_MODELS = os.path.join(BASE_DIR, 'models')

# ═══════════════════════════════════════════════════════
#  STUDENT MODELS
# ═══════════════════════════════════════════════════════
print("\n── Retraining Student models ──────────────────────")

df = pd.read_csv(STUDENT_DATA)

tier_map = {'Tier 1': 3, 'Tier 2': 2, 'Tier 3': 1}
df['college_tier_encoded'] = df['college_tier'].map(tier_map)

le = LabelEncoder()
df['job_role_encoded'] = le.fit_transform(df['job_role'])

ALL_SKILLS = [
    'DSA', 'Machine Learning', 'Deep Learning', 'TensorFlow', 'PyTorch',
    'Python', 'Java', 'React', 'AWS', 'Docker', 'Node.js', 'Django',
    'Excel', 'SQL', 'HTML/CSS', 'JavaScript', 'Git', 'Linux'
]

SALARY_FEATURES = [
    'college_tier_encoded', 'cgpa', 'internships', 'github_projects',
    'backlogs', 'hackathons', 'certifications', 'aptitude_score',
    'advanced_skills', 'intermediate_skills', 'basic_skills'
]

df['advanced_skills']     = df['advanced_skills'].fillna(0)
df['intermediate_skills'] = df['intermediate_skills'].fillna(0)
df['basic_skills']        = df['basic_skills'].fillna(0)

X_sal = df[SALARY_FEATURES].fillna(0)
y_sal = df['salary_lpa']

salary_model = RandomForestRegressor(n_estimators=100, random_state=42)
salary_model.fit(X_sal, y_sal)
sal_r2 = round(r2_score(y_sal, salary_model.predict(X_sal)), 4)
print(f"   Salary model trained  — R²={sal_r2}")

role_base = [
    'college_tier_encoded', 'cgpa', 'internships', 'github_projects',
    'backlogs', 'hackathons', 'certifications', 'aptitude_score'
]
skill_cols = []
for s in ALL_SKILLS:
    col = 'skill_' + s.lower().replace('/', '_').replace('.', '_').replace(' ', '_')
    df[col] = 0
    skill_cols.append(col)

ROLE_FEATURES = role_base + skill_cols
X_role = df[ROLE_FEATURES].fillna(0)
y_role = df['job_role_encoded']

role_model = XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss')
role_model.fit(X_role, y_role)
role_acc = round(accuracy_score(y_role, role_model.predict(X_role)), 4)
print(f"   Role model trained    — Acc={role_acc*100:.1f}%")

os.makedirs(STUDENT_MODELS, exist_ok=True)
with open(os.path.join(STUDENT_MODELS, 'salary_model.pkl'), 'wb') as f:
    pickle.dump(salary_model, f)
with open(os.path.join(STUDENT_MODELS, 'role_model.pkl'), 'wb') as f:
    pickle.dump(role_model, f)
with open(os.path.join(STUDENT_MODELS, 'label_encoder_role.pkl'), 'wb') as f:
    pickle.dump(le, f)

meta = {
    'salary_model':    'Random Forest',
    'salary_r2':       sal_r2,
    'role_model':      'XGBoost',
    'role_accuracy':   role_acc,
    'salary_features': SALARY_FEATURES,
    'role_features':   ROLE_FEATURES,
    'all_skills':      ALL_SKILLS,
    'tier_mapping':    tier_map
}
with open(os.path.join(STUDENT_MODELS, 'feature_columns.json'), 'w') as f:
    json.dump(meta, f, indent=2)
print("   Student models saved ✓")

# ═══════════════════════════════════════════════════════
#  RECRUITER MODELS
#  IMPORTANT: Feature columns here must EXACTLY match
#  what ranker.py fills into feature_row{}
# ═══════════════════════════════════════════════════════
print("\n── Retraining Recruiter models ─────────────────────")

df2 = pd.read_csv(RECRUITER_DATA)

# These EXACTLY match what ranker.py fills in feature_row{}
# If you change ranker.py, update this list too
FEATURE_COLS = [
    'total_experience_yrs', 'expected_salary_inr', 'highest_cgpa',
    'total_skills_count', 'num_projects', 'num_certifications',
    'num_companies', 'highest_degree_rank', 'institute_tier', 'has_certification',
    'skill_python', 'skill_java', 'skill_cplusplus', 'skill_javascript',
    'skill_typescript', 'skill_go', 'skill_r', 'skill_react', 'skill_angular',
    'skill_vuejs', 'skill_nodejs', 'skill_django', 'skill_flask', 'skill_fastapi',
    'skill_spring_boot', 'skill_nextjs', 'skill_html', 'skill_css',
    'skill_pandas', 'skill_numpy', 'skill_scikit_learn', 'skill_tensorflow',
    'skill_pytorch', 'skill_keras', 'skill_xgboost', 'skill_tableau',
    'skill_power_bi', 'skill_mysql', 'skill_postgresql', 'skill_mongodb',
    'skill_redis', 'skill_aws', 'skill_azure', 'skill_gcp', 'skill_docker',
    'skill_kubernetes', 'skill_git', 'skill_linux', 'skill_bash',
    'skill_react_native', 'skill_flutter', 'skill_rest_api', 'skill_graphql',
    'skill_microservices', 'skill_cicd', 'skill_agile', 'skill_scrum',
    'skill_system_design', 'skill_data_structures', 'skill_algorithms',
]

# Remove duplicates preserving order
seen = set()
FEATURE_COLS = [x for x in FEATURE_COLS if not (x in seen or seen.add(x))]
# Keep only cols that exist in CSV
FEATURE_COLS = [c for c in FEATURE_COLS if c in df2.columns]
print(f"   Using {len(FEATURE_COLS)} features")

X2 = df2[FEATURE_COLS].fillna(0)
y2 = df2['candidate_score'].fillna(0)

scaler = StandardScaler()
X2_scaled = scaler.fit_transform(X2)

rec_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
rec_model.fit(X2_scaled, y2)
rec_r2 = round(r2_score(y2, rec_model.predict(X2_scaled)), 4)
print(f"   Ranking model trained — R²={rec_r2}")

os.makedirs(RECRUITER_MODELS, exist_ok=True)
joblib.dump(rec_model,    os.path.join(RECRUITER_MODELS, 'ranking_model.pkl'))
joblib.dump(scaler,       os.path.join(RECRUITER_MODELS, 'scaler.pkl'))
joblib.dump(FEATURE_COLS, os.path.join(RECRUITER_MODELS, 'feature_columns.pkl'))
print("   Recruiter models saved ✓")

print("\n✅ ALL MODELS RETRAINED SUCCESSFULLY")
print("   Now run: python app.py")