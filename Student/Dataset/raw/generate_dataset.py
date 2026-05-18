import pandas as pd
import numpy as np
import random
import os
 
np.random.seed(42)
N = 5000
 
# ─── CONFIG ──────────────────────────────────────────────────────────────────
 
TIER_BOOST   = {"Tier 1": 1.55, "Tier 2": 1.10, "Tier 3": 0.80}
TIER_WEIGHTS = [0.20, 0.45, 0.35]
TIERS        = list(TIER_BOOST.keys())
 
ADVANCED     = {"DSA", "Machine Learning", "Deep Learning", "TensorFlow", "PyTorch"}
INTERMEDIATE = {"Python", "Java", "React", "AWS", "Docker", "Node.js", "Django"}
BASIC        = {"Excel", "SQL", "HTML/CSS", "JavaScript", "Git", "Linux"}
ALL_SKILLS   = list(ADVANCED | INTERMEDIATE | BASIC)
 
# Skills are intentionally shared across roles (Python appears in 7 roles,
# SQL in 3, Java in 3, etc.) so that the ML model faces genuine ambiguity
# and must learn real patterns rather than memorise a lookup table.
JOB_ROLES = {
    "ML Engineer":       {"min_cgpa": 7.5, "skills": {"Machine Learning","Deep Learning","TensorFlow","PyTorch","Python"},   "band": (10, 28)},
    "Data Scientist":    {"min_cgpa": 7.0, "skills": {"Machine Learning","Deep Learning","Python","TensorFlow","SQL"},        "band": (8,  22)},
    "Software Engineer": {"min_cgpa": 6.0, "skills": {"DSA","Python","Java","React","Git"},                                  "band": (5,  20)},
    "Backend Developer": {"min_cgpa": 6.0, "skills": {"Java","Python","Docker","SQL","Django"},                              "band": (5,  16)},
    "Web Developer":     {"min_cgpa": 5.5, "skills": {"React","JavaScript","HTML/CSS","Node.js","Django"},                   "band": (4,  14)},
    "DevOps Engineer":   {"min_cgpa": 6.5, "skills": {"Docker","AWS","Linux","Python","Git"},                                "band": (7,  18)},
    "Data Analyst":      {"min_cgpa": 6.0, "skills": {"SQL","Excel","Python","JavaScript"},                                  "band": (4,  12)},
    "Junior Developer":  {"min_cgpa": 4.0, "skills": {"Python","JavaScript","HTML/CSS","Git","Java"},                        "band": (3,   8)},
}
 
# ─── FUNCTIONS ───────────────────────────────────────────────────────────────
 
def assign_job_role(skills, cgpa):
    """
    Assigns a job role based on skill overlap count with each role's skill set.
 
    No if/else rules for specific skills — the function does not say
    "if TensorFlow then ML Engineer". It counts how many of the student's
    skills appear in each role's skill set and picks the best match.
 
    When two roles tie on overlap count, the higher CGPA wins. Remaining
    ties are broken alphabetically for stability (reproducibility).
 
    Ambiguity is built into the data structure itself: Python appears in
    7 different roles, SQL in 3, Java in 3 — so a student with only
    Python is genuinely hard to classify, just like in real hiring.
    Expected model accuracy: 85–90% (realistic for an 8-class problem).
    """
    skill_set = set(skills)
    scored    = []
 
    for role, info in JOB_ROLES.items():
        if cgpa < info["min_cgpa"]:
            continue
        if cgpa >= 7.5 and role == "Junior Developer":
            continue
        overlap = len(skill_set & info["skills"])
        if overlap > 0:
            scored.append((role, overlap, cgpa))
 
    if not scored:
        return "Junior Developer"
 
    # Sort: most overlapping skills first → higher CGPA as tiebreaker → alphabetical
    scored.sort(key=lambda x: (-x[1], -x[2], x[0]))
    return scored[0][0]
 
 
def generate_aptitude(cgpa, adv):
    """
    Simulates the in-app aptitude test score (0–100).
    Correlated with CGPA and advanced skill count, with realistic noise.
    """
    score = cgpa * 7 + adv * 4 + np.random.normal(0, 6)
    return round(float(np.clip(score, 10, 100)), 1)
 
 
def calculate_salary(cgpa, adv, inter, basic, internships, github_projects,
                     aptitude, backlogs, hackathons, certifications, tier, band):
 
    salary = 2.5
    salary += (cgpa - 5) * 1.6
    salary += adv   * 3.2
    salary += inter * 1.4
    salary += basic * 0.4
    salary += internships * 1.0
    salary += min(github_projects * 0.8, 3.0)
    salary += (aptitude / 100) * 3.0
    salary += hackathons     * 0.5
    salary += certifications * 0.4
    salary -= backlogs * random.uniform(0.7, 1.2)
 
    if cgpa > 8.5:
        salary += random.uniform(2, 4)
 
    if cgpa >= 6:
        salary *= TIER_BOOST[tier]
 
    if   cgpa < 5.5: salary = min(salary, 4.5)
    elif cgpa < 6.0: salary = min(salary, 6.0)
    elif cgpa < 7.0: salary = min(salary, 10.0)
    elif cgpa < 8.0: salary = min(salary, 16.0)
    elif cgpa < 9.0: salary = min(salary, 24.0)
 
    if adv == 0:                salary = min(salary, 13.0)
    if adv == 0 and inter == 0: salary = min(salary, 8.0)
    if adv == 0:                salary *= 0.8
 
    salary = max(band[0] * 0.7, min(salary, band[1] * 1.1))
    salary += np.random.normal(0, 0.3)
    return round(max(2.0, min(float(salary), 45.0)), 2)
 
 
def generate_student():
    tier           = random.choices(TIERS, weights=TIER_WEIGHTS)[0]
    cgpa           = round(np.clip(np.random.normal(7, 1.2), 4, 10), 2)
    internships    = int(np.random.poisson(1))
    github_projects = int(np.random.poisson(2))
    backlogs       = int(np.random.poisson(0.5 if cgpa > 7 else 1))
    hackathons     = int(np.clip(np.random.poisson(1.5), 0, 8))
    certifications = int(np.clip(np.random.poisson(2),   0, 8))
 
    num_skills = random.randint(1, min(6, int(cgpa // 2) + 2))
    skills     = random.sample(ALL_SKILLS, num_skills)
    skill_set  = set(skills)
 
    adv   = len(skill_set & ADVANCED)
    inter = len(skill_set & INTERMEDIATE)
    basic = len(skill_set & BASIC)
 
    aptitude = generate_aptitude(cgpa, adv)
    role     = assign_job_role(skills, cgpa)
    band     = JOB_ROLES[role]["band"]
 
    salary = calculate_salary(
        cgpa, adv, inter, basic,
        internships, github_projects,
        aptitude, backlogs,
        hackathons, certifications,
        tier, band
    )
 
    return {
        "college_tier":        tier,
        "cgpa":                cgpa,
        "internships":         internships,
        "github_projects":     github_projects,
        "backlogs":            backlogs,
        "hackathons":          hackathons,
        "certifications":      certifications,
        "skills":              ", ".join(skills),
        "advanced_skills":     adv,
        "intermediate_skills": inter,
        "basic_skills":        basic,
        "aptitude_score":      aptitude,
        "job_role":            role,
        "salary_lpa":          salary,
    }


# ─── GENERATE DATASET ──────────────────────────────────

data = pd.DataFrame([generate_student() for _ in range(N)])

data.to_csv("Student/Dataset/raw/student_salary_dataset.csv", index=False)

print("✅ Dataset generated successfully!")
print(data.head())