"""
utils/ranker.py
----------------
Handles ranking logic for uploaded resumes using the ML model.
Simple functions — one job each.
"""

import numpy as np
import pandas as pd


SCALE_COLS = [
    "total_experience_yrs",
    "expected_salary_inr",
    "highest_cgpa",
    "total_skills_count",
    "num_projects",
    "num_certifications",
    "num_companies",
]


def skill_to_col(skill):
    """Convert skill name to CSV column name."""
    return (
        "skill_"
        + skill.lower()
        .replace(" ", "_")
        .replace(".", "")
        .replace("+", "plus")
        .replace("/", "")
        .replace("-", "_")
    )


def rank_uploaded_resumes(parsed_list, required_skills, model, scaler, feature_cols):
    """
    Score and rank a list of parsed resumes using the ML model.

    Steps:
      1. Build a feature row for each resume
      2. Scale continuous columns
      3. Predict score using ML model
      4. Combine with skill match percentage
      5. Sort by final score

    Returns a sorted list of result dicts.
    """
    results = []

    for parsed in parsed_list:
        if not parsed:
            continue

        # Skill match
        resume_skills  = [s.lower() for s in parsed.get("skills", [])]
        matched_skills = [s for s in required_skills if s.lower() in resume_skills]
        total_required = max(len(required_skills), 1)
        match_pct      = round(len(matched_skills) / total_required * 100, 1)

        # Build feature row — start with zeros
        feature_row = {col: 0 for col in feature_cols}

        # Fill in known values
        feature_row["total_experience_yrs"] = parsed.get("experience_yrs", 0)
        feature_row["total_skills_count"]   = len(parsed.get("skills", []))
        feature_row["num_companies"]        = len(parsed.get("companies", []))
        feature_row["num_certifications"]   = 0
        feature_row["num_projects"]         = 0
        feature_row["highest_cgpa"]         = 7.0   # default average
        feature_row["highest_degree_rank"]  = 5     # default B.Tech
        feature_row["institute_tier"]       = 2     # default tier 2
        feature_row["role_encoded"]         = 0
        feature_row["has_certification"]    = 0
        feature_row["expected_salary_inr"]  = 1000000  # default 10L

        # Fill skill binary flags
        for skill in parsed.get("skills", []):
            col = skill_to_col(skill)
            if col in feature_row:
                feature_row[col] = 1
# Scale and predict
            row_df = pd.DataFrame([feature_row])[feature_cols]
            row_df = row_df.fillna(0)
            row_scaled = scaler.transform(row_df)

# Predict
        ml_score = float(model.predict(row_scaled)[0])
        ml_score   = round(max(0, min(100, ml_score)), 2)

        # Final score: 60% ML + 40% skill match
        final_score = round(0.6 * ml_score + 0.4 * match_pct, 2)

        results.append({
            "filename"       : parsed.get("filename", ""),
            "name"           : parsed.get("name", "Unknown"),
            "email"          : parsed.get("email", "—"),
            "phone"          : parsed.get("phone", "—"),
            "experience"     : parsed.get("experience_yrs", 0),
            "college"        : parsed.get("college", "—"),
            "degree"         : parsed.get("degree", "—"),
            "designation"    : parsed.get("designation", "—"),
            "companies"      : parsed.get("companies", []),
            "skills"         : parsed.get("skills", []),
            "skills_matched" : matched_skills,
            "match_pct"      : match_pct,
            "ml_score"       : ml_score,
            "final_score"    : final_score,
        })

    # Sort by final score descending
    results = sorted(results, key=lambda x: x["final_score"], reverse=True)

    # Add rank
    for i, r in enumerate(results, 1):
        r["rank"] = i

    return results