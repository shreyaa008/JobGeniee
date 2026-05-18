"""
step2_generate_ranking_dataset.py
-----------------------------------
Generates ONE raw CSV file: candidates.csv

  - Real human-readable values (salary in INR, years, CGPA)
  - Has missing values + duplicate rows for cleaning practice
  - Cleaning, scaling, training all happen in step4_train_ranking_model.ipynb

Run from project root:
  python notebooks/step2_generate_ranking_dataset.py
"""

import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, CANDIDATES_CSV

np.random.seed(42)
os.makedirs(DATA_DIR, exist_ok=True)

N = 5000

# ── Roles ─────────────────────────────────────────────────────────────────────

ROLES = [
    "Backend Developer", "Frontend Developer", "Data Scientist",
    "DevOps Engineer", "Full Stack Developer", "ML Engineer",
    "iOS Developer", "Android Developer",
]

ALL_SKILLS = [
    "python","java","cplusplus","c","javascript","typescript","go","rust",
    "kotlin","swift","r","scala","php","react","angular","vuejs","nodejs",
    "django","flask","fastapi","spring_boot","expressjs","nextjs","html","css",
    "pandas","numpy","scikit_learn","tensorflow","pytorch","keras","xgboost",
    "matplotlib","seaborn","tableau","power_bi","mysql","postgresql","mongodb",
    "redis","sqlite","cassandra","oracle_db","firebase","aws","azure","gcp",
    "docker","kubernetes","jenkins","git","github_actions","terraform","linux",
    "bash","react_native","flutter","android_sdk","ios_sdk","rest_api","graphql",
    "microservices","cicd","agile","scrum","system_design","data_structures",
    "algorithms",
]

ROLE_SKILL_PROBS = {
    "Backend Developer": {
        "python":0.85,"java":0.60,"django":0.65,"flask":0.55,"fastapi":0.45,
        "spring_boot":0.40,"rest_api":0.90,"microservices":0.55,
        "postgresql":0.70,"mysql":0.60,"mongodb":0.50,"redis":0.45,
        "docker":0.65,"kubernetes":0.40,"jenkins":0.45,"git":0.95,
        "linux":0.70,"bash":0.55,"aws":0.55,"cicd":0.50,
        "agile":0.60,"scrum":0.55,"system_design":0.65,
        "data_structures":0.70,"algorithms":0.70,
    },
    "Frontend Developer": {
        "javascript":0.95,"typescript":0.70,"react":0.75,"angular":0.45,
        "vuejs":0.35,"nodejs":0.55,"nextjs":0.50,"html":0.98,"css":0.98,
        "expressjs":0.40,"mongodb":0.35,"git":0.90,"rest_api":0.70,
        "graphql":0.35,"agile":0.60,"scrum":0.55,"firebase":0.40,
        "docker":0.30,"aws":0.25,"data_structures":0.45,"algorithms":0.45,
    },
    "Data Scientist": {
        "python":0.97,"r":0.55,"pandas":0.95,"numpy":0.93,
        "scikit_learn":0.90,"tensorflow":0.65,"pytorch":0.60,"keras":0.55,
        "xgboost":0.70,"matplotlib":0.88,"seaborn":0.82,"tableau":0.50,
        "power_bi":0.40,"mysql":0.60,"postgresql":0.50,"aws":0.45,
        "git":0.85,"docker":0.40,"agile":0.55,
        "data_structures":0.60,"algorithms":0.65,
    },
    "DevOps Engineer": {
        "python":0.65,"bash":0.90,"linux":0.95,"docker":0.92,
        "kubernetes":0.85,"jenkins":0.80,"terraform":0.70,"aws":0.80,
        "azure":0.55,"gcp":0.45,"github_actions":0.65,"cicd":0.90,
        "git":0.95,"agile":0.55,"scrum":0.50,"system_design":0.60,"go":0.25,
    },
    "Full Stack Developer": {
        "javascript":0.90,"typescript":0.60,"react":0.70,"nodejs":0.75,
        "python":0.55,"html":0.95,"css":0.95,"nextjs":0.45,"expressjs":0.55,
        "django":0.35,"flask":0.30,"mongodb":0.60,"postgresql":0.55,
        "mysql":0.50,"redis":0.40,"docker":0.55,"aws":0.45,"git":0.92,
        "rest_api":0.80,"graphql":0.40,"agile":0.65,"scrum":0.60,
        "system_design":0.55,"data_structures":0.55,"algorithms":0.55,
    },
    "ML Engineer": {
        "python":0.98,"tensorflow":0.80,"pytorch":0.75,"keras":0.65,
        "scikit_learn":0.85,"pandas":0.90,"numpy":0.92,"xgboost":0.65,
        "docker":0.70,"kubernetes":0.55,"aws":0.60,"gcp":0.45,"git":0.88,
        "postgresql":0.45,"bash":0.60,"linux":0.65,"cicd":0.55,
        "agile":0.55,"data_structures":0.70,"algorithms":0.75,
    },
    "iOS Developer": {
        "swift":0.95,"ios_sdk":0.95,"react_native":0.30,"flutter":0.25,
        "git":0.90,"firebase":0.55,"rest_api":0.75,"sqlite":0.50,
        "aws":0.30,"agile":0.65,"scrum":0.60,
        "data_structures":0.65,"algorithms":0.65,
    },
    "Android Developer": {
        "kotlin":0.90,"java":0.80,"android_sdk":0.95,"react_native":0.25,
        "flutter":0.30,"git":0.90,"firebase":0.60,"rest_api":0.75,
        "sqlite":0.55,"aws":0.30,"agile":0.65,"scrum":0.60,
        "data_structures":0.65,"algorithms":0.65,
    },
}

# ── Generator functions ───────────────────────────────────────────────────────

def gen_experience():
    buckets = [0,  1,  2,  3,  5,  8,  12]
    weights = [0.20,0.18,0.16,0.14,0.14,0.10,0.08]
    base  = np.random.choice(buckets, p=weights)
    exp   = round(base + np.random.uniform(0, 0.9), 1)
    if np.random.random() < 0.03:
        exp = round(np.random.uniform(15, 25), 1)
    return exp

def gen_salary(exp, role):
    ranges = {
        "Backend Developer":    (350000, 2500000),
        "Frontend Developer":   (300000, 2000000),
        "Data Scientist":       (400000, 2800000),
        "DevOps Engineer":      (400000, 2600000),
        "Full Stack Developer": (350000, 2300000),
        "ML Engineer":          (500000, 3200000),
        "iOS Developer":        (350000, 2200000),
        "Android Developer":    (350000, 2200000),
    }
    lo, hi = ranges[role]
    center = lo + (hi - lo) * (min(exp / 12, 1.0) ** 0.7)
    salary = int(round(max(lo * 0.8, center + np.random.normal(0, center * 0.12)) / 10000) * 10000)
    if np.random.random() < 0.02:
        salary = int(salary * np.random.uniform(1.5, 2.2))
    return salary

def gen_cgpa(degree_rank, tier):
    mu = {1:8.2, 2:7.8, 3:7.5, 4:7.2, 5:7.0, 6:6.5}.get(degree_rank, 7.0)
    mu += 0.3 if tier == 1 else (-0.4 if tier == 3 else 0)
    cgpa = round(max(4.0, min(10.0, np.random.normal(mu, 0.8))), 2)
    if np.random.random() < 0.05: cgpa = round(np.random.uniform(4.0, 5.5), 2)
    if np.random.random() < 0.04: cgpa = round(np.random.uniform(9.2, 10.0), 2)
    return cgpa

def gen_skills(role, exp):
    probs = ROLE_SKILL_PROBS.get(role, {})
    flags = {}
    for skill in ALL_SKILLS:
        prob = min(probs.get(skill, 0.05) + exp * 0.015, 0.97)
        flags[f"skill_{skill}"] = int(np.random.random() < prob)
    return flags

def compute_score(exp, cgpa, degree_rank, tier, num_skills, num_certs, num_projects, num_companies):
    """Target label — computed from rubric, NOT from training features."""
    s = 0.0
    # Experience
    if exp == 0:       s += 5
    elif exp <= 2:     s += 10 + exp * 3
    elif exp <= 8:     s += 18 + (exp - 2) * 1.5
    else:              s += 27 - (exp - 8) * 0.3
    # CGPA
    s += 15 if cgpa>=9 else 12 if cgpa>=8 else 9 if cgpa>=7 else 6 if cgpa>=6 else 3
    # Institute tier
    s += {1:10, 2:7, 3:4}.get(tier, 4)
    # Degree
    s += {1:8, 2:7, 3:6, 4:5, 5:4, 6:2}.get(degree_rank, 4)
    # Other
    s += min(num_skills * 0.5, 20)
    s += min(num_projects * 2.5, 10)
    s += min(num_certs * 2.5, 8)
    s += min(num_companies * 1.5, 4)
    s += np.random.normal(0, 2.5)
    return round(max(0.0, min(100.0, s)), 2)

# ── Build dataset ─────────────────────────────────────────────────────────────

print("=" * 50)
print("Step 2 — Generating ranking dataset (1 CSV)")
print("=" * 50)

DEGREE_WEIGHTS = [0.05, 0.12, 0.08, 0.08, 0.60, 0.07]

rows = []
for i in range(N):
    role   = np.random.choice(ROLES)
    exp    = gen_experience()
    salary = gen_salary(exp, role)
    deg    = int(np.random.choice([1,2,3,4,5,6], p=DEGREE_WEIGHTS))
    tier   = int(np.random.choice([1,2,3], p=[0.15,0.40,0.45]))
    cgpa   = gen_cgpa(deg, tier)
    skills = gen_skills(role, exp)
    ns     = sum(skills.values())
    cert   = int(np.random.random() < min(0.2 + exp*0.04, 0.85))
    nc     = int(np.random.poisson(1.2)) if cert else 0
    np_    = max(0, int(np.random.poisson(max(1.5, 3.5 - exp*0.1))))
    nco    = max(0, min(int(exp/2.5) + np.random.randint(-1,2), 8)) if exp>0 else 0

    rows.append({
        "total_experience_yrs" : exp,
        "expected_salary_inr"  : salary,
        "highest_degree_rank"  : deg,
        "highest_cgpa"         : cgpa,
        "institute_tier"       : tier,
        **skills,
        "total_skills_count"   : ns,
        "has_certification"    : cert,
        "num_certifications"   : nc,
        "num_projects"         : np_,
        "num_companies"        : nco,
        "role_applied_for"     : role,
        "candidate_score"      : compute_score(exp, cgpa, deg, tier, ns, nc, np_, nco),
    })

df = pd.DataFrame(rows)
print(f"Clean rows : {len(df)}")

# ── Inject missing values + duplicates ────────────────────────────────────────

MISSING = {
    "highest_cgpa":0.08, "num_certifications":0.10, "num_projects":0.07,
    "num_companies":0.05, "institute_tier":0.06,
    "expected_salary_inr":0.04, "highest_degree_rank":0.03,
}
for col, rate in MISSING.items():
    df.loc[np.random.random(len(df)) < rate, col] = np.nan

# Bad salary entries
bad = np.random.choice(df.index, 15, replace=False)
df.loc[bad, "expected_salary_inr"] = np.random.choice([0,-1,999999999], 15)

# Duplicate rows (~1%)
dups = np.random.choice(df.index, 50, replace=False)
df   = pd.concat([df, df.loc[dups]], ignore_index=True)

print(f"Total rows (with duplicates + missing): {len(df)}")
print("Missing values per column:")
miss = df.isnull().sum()
for col, cnt in miss[miss>0].items():
    print(f"  {col:30s} {cnt}")

# ── Save ──────────────────────────────────────────────────────────────────────

df.to_csv(CANDIDATES_CSV, index=False)
print(f"\nSaved: {CANDIDATES_CSV}")

print("\n── Stats ─────────────────────────────────────────────")
clean = df.dropna()
print(f"Experience : {df.total_experience_yrs.min():.1f} - {df.total_experience_yrs.max():.1f} yrs")
print(f"Salary     : Rs{int(df.expected_salary_inr.min()):,} - Rs{int(df.expected_salary_inr.max()):,}")
print(f"Score      : {df.candidate_score.min():.1f} - {df.candidate_score.max():.1f}  (mean {df.candidate_score.mean():.1f})")
print(f"CGPA       : {df.highest_cgpa.min():.1f} - {df.highest_cgpa.max():.1f}")
print("\nRole counts:")
for role, cnt in df.role_applied_for.value_counts().items():
    print(f"  {role:25s} {cnt}")

print("\nStep 2 complete! Next: open step3_train_ner_model.ipynb")