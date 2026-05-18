"""
step1_generate_ner_dataset.py
------------------------------
Generates NER training data in spaCy-compatible format.

Each sample:
  {
    "text": "John Doe\njohn@email.com\n...",
    "entities": [[0, 8, "Name"], [9, 25, "Email"], ...]
  }

Run from project root:
  python notebooks/step1_generate_ner_dataset.py
"""

import json
import random
import os
import sys
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import NER_DATA_PATH, DATA_DIR

random.seed(42)
os.makedirs(DATA_DIR, exist_ok=True)

# ── Data pools ────────────────────────────────────────────────────────────────

FIRST_NAMES = [
    "Aarav", "Arjun", "Rohan", "Vikram", "Rahul", "Amit", "Karan", "Nikhil",
    "Siddharth", "Priya", "Ananya", "Sneha", "Pooja", "Divya", "Meera",
    "Tanishq", "Shreya", "Riya", "Nisha", "Kavya", "Harsh", "Yash", "Dev",
    "Ishaan", "Aditya", "Varun", "Saurabh", "Ritesh", "Mohit", "Gaurav",
]

LAST_NAMES = [
    "Sharma", "Verma", "Singh", "Kumar", "Gupta", "Joshi", "Patel", "Shah",
    "Mehta", "Agarwal", "Nair", "Iyer", "Reddy", "Rao", "Pillai", "Chopra",
    "Malhotra", "Bose", "Das", "Mukherjee", "Pandey", "Mishra", "Tiwari",
    "Chauhan", "Yadav", "Garg", "Kapoor", "Bhatia", "Sinha", "Khanna",
]

EMAIL_DOMAINS = ["gmail.com", "yahoo.com", "outlook.com", "hotmail.com"]

PHONES = [
    "9812345670", "8987654321", "7654321098", "9001234567", "8123456789",
    "9876543210", "7890123456", "9123456780", "8012345678", "9988776655",
    "7011223344", "9517284630", "8826374819", "9731082645", "8190263748",
]

CITIES = [
    "Bangalore", "Mumbai", "Delhi", "Hyderabad", "Pune", "Chennai",
    "Kolkata", "Ahmedabad", "Jaipur", "Noida", "Gurgaon", "Chandigarh",
]

COLLEGES = [
    "IIT Bombay", "IIT Delhi", "IIT Madras", "IIT Kharagpur",
    "NIT Trichy", "NIT Warangal", "NIT Surathkal",
    "BITS Pilani", "VIT Vellore", "Anna University",
    "Delhi University", "Pune University", "Mumbai University",
    "Manipal Institute of Technology", "SRM University",
]

DEGREES = [
    "B.Tech in Computer Science",
    "B.Tech in Information Technology",
    "B.Tech in Electronics and Communication",
    "M.Tech in Computer Science",
    "MCA",
    "B.Sc in Computer Science",
    "M.Sc in Data Science",
]

COMPANIES = [
    "TCS", "Infosys", "Wipro", "HCL", "Tech Mahindra",
    "Accenture", "IBM", "Cognizant", "Capgemini", "Deloitte",
    "Amazon", "Microsoft", "Google", "Flipkart", "Zoho",
    "Freshworks", "Razorpay", "PhonePe", "Swiggy", "Byjus",
]

DESIGNATIONS = [
    "Software Engineer", "Senior Software Engineer", "Junior Developer",
    "Backend Developer", "Frontend Developer", "Full Stack Developer",
    "Data Scientist", "ML Engineer", "DevOps Engineer",
    "Software Developer", "Associate Engineer", "Tech Lead",
]

SKILL_GROUPS = {
    "languages"  : ["Python", "Java", "JavaScript", "TypeScript", "Go", "Swift", "Kotlin", "R", "SQL"],
    "frameworks" : ["Django", "Flask", "FastAPI", "React", "Angular", "Node.js", "Spring Boot", "Next.js"],
    "ml"         : ["TensorFlow", "PyTorch", "Scikit-learn", "Keras", "XGBoost", "Pandas", "NumPy"],
    "databases"  : ["MySQL", "PostgreSQL", "MongoDB", "Redis", "SQLite", "Firebase"],
    "cloud"      : ["AWS", "Azure", "GCP", "Docker", "Kubernetes", "Jenkins", "Git", "Linux"],
    "concepts"   : ["REST API", "GraphQL", "Microservices", "Agile", "Scrum", "System Design"],
}

CERTIFICATIONS = [
    "AWS Certified Solutions Architect",
    "Google Cloud Professional Data Engineer",
    "Microsoft Azure Fundamentals",
    "Docker Certified Associate",
    "Certified Kubernetes Administrator",
    "TensorFlow Developer Certificate",
    "Oracle Java SE Programmer",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def random_name():
    return f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"

def random_email(name):
    parts = name.lower().split()
    patterns = [
        f"{parts[0]}.{parts[1]}",
        f"{parts[0]}{parts[1][0]}",
        f"{parts[0]}_{parts[1]}",
        f"{parts[0]}{random.randint(1, 99)}",
    ]
    return f"{random.choice(patterns)}@{random.choice(EMAIL_DOMAINS)}"

def random_skills(n=6):
    picked = []
    groups = random.sample(list(SKILL_GROUPS.keys()), k=min(n, len(SKILL_GROUPS)))
    for group in groups:
        picked.append(random.choice(SKILL_GROUPS[group]))
    all_skills = [s for g in SKILL_GROUPS.values() for s in g]
    while len(picked) < n:
        s = random.choice(all_skills)
        if s not in picked:
            picked.append(s)
    return picked

def random_year(min_yr=2010, max_yr=2023):
    return str(random.randint(min_yr, max_yr))

def find_span(text, substring):
    idx = text.find(substring)
    if idx == -1:
        return None
    return (idx, idx + len(substring))

def add_entity(entities, text, value, label):
    span = find_span(text, value)
    if span:
        entities.append([span[0], span[1], label])

def remove_overlapping(entities):
    """spaCy crashes on overlapping spans — keep only non-overlapping."""
    entities = sorted(entities, key=lambda x: x[0])
    clean = []
    last_end = -1
    for start, end, label in entities:
        if start >= last_end:
            clean.append([start, end, label])
            last_end = end
    return clean

# ── Resume templates ──────────────────────────────────────────────────────────

def build_v1_standard():
    """Standard format: Name at top, sections below."""
    name        = random_name()
    email       = random_email(name)
    phone       = random.choice(PHONES)
    city        = random.choice(CITIES)
    college     = random.choice(COLLEGES)
    degree      = random.choice(DEGREES)
    grad_year   = random_year(2015, 2023)
    company1    = random.choice(COMPANIES)
    company2    = random.choice([c for c in COMPANIES if c != company1])
    designation = random.choice(DESIGNATIONS)
    skills      = random_skills(random.randint(6, 10))
    skills_str  = ", ".join(skills)
    cert        = random.choice(CERTIFICATIONS)
    exp_yrs     = str(random.randint(1, 8))

    text = (
        f"{name}\n"
        f"{email} | {phone} | {city}\n\n"
        f"EDUCATION\n"
        f"{degree} — {college}\n"
        f"Graduation Year: {grad_year}\n\n"
        f"EXPERIENCE\n"
        f"{designation} — {company1}\n"
        f"{exp_yrs} years of experience in software development.\n\n"
        f"Software Developer — {company2}\n"
        f"2 years\n\n"
        f"SKILLS\n"
        f"{skills_str}\n\n"
        f"CERTIFICATIONS\n"
        f"{cert}\n"
    )

    entities = []
    add_entity(entities, text, name,        "Name")
    add_entity(entities, text, email,       "Email")
    add_entity(entities, text, phone,       "Phone")
    add_entity(entities, text, college,     "College")
    add_entity(entities, text, degree,      "Degree")
    add_entity(entities, text, grad_year,   "Graduation_Year")
    add_entity(entities, text, company1,    "Company")
    add_entity(entities, text, company2,    "Company")
    add_entity(entities, text, designation, "Designation")
    add_entity(entities, text, cert,        "Certification")
    for skill in skills:
        add_entity(entities, text, skill, "Skill")

    return text.strip(), entities


def build_v2_compact():
    """Compact format with bullet-point skills."""
    name        = random_name()
    email       = random_email(name)
    phone       = random.choice(PHONES)
    city        = random.choice(CITIES)
    college     = random.choice(COLLEGES)
    degree      = random.choice(DEGREES)
    grad_year   = random_year(2015, 2023)
    company     = random.choice(COMPANIES)
    designation = random.choice(DESIGNATIONS)
    skills      = random_skills(random.randint(5, 9))
    exp_yrs     = str(random.randint(1, 6))

    skill_bullets = "\n".join([f"* {s}" for s in skills])

    text = (
        f"{name}\n"
        f"Location: {city} | Email: {email} | Mobile: {phone}\n\n"
        f"Summary\n"
        f"{exp_yrs}+ years of experience as {designation}.\n\n"
        f"Technical Skills\n"
        f"{skill_bullets}\n\n"
        f"Work Experience\n"
        f"{designation} at {company}\n\n"
        f"Education\n"
        f"{degree}\n"
        f"{college} — {grad_year}\n"
    )

    entities = []
    add_entity(entities, text, name,        "Name")
    add_entity(entities, text, email,       "Email")
    add_entity(entities, text, phone,       "Phone")
    add_entity(entities, text, college,     "College")
    add_entity(entities, text, degree,      "Degree")
    add_entity(entities, text, grad_year,   "Graduation_Year")
    add_entity(entities, text, company,     "Company")
    add_entity(entities, text, designation, "Designation")
    for skill in skills:
        add_entity(entities, text, skill, "Skill")

    return text.strip(), entities


def build_v3_fresher():
    """Fresher resume — no work experience."""
    name      = random_name()
    email     = random_email(name)
    phone     = random.choice(PHONES)
    city      = random.choice(CITIES)
    college   = random.choice(COLLEGES)
    degree    = random.choice(DEGREES)
    grad_year = random_year(2022, 2024)
    skills    = random_skills(random.randint(4, 8))
    skills_str= ", ".join(skills)

    text = (
        f"CURRICULUM VITAE\n\n"
        f"Name: {name}\n"
        f"Contact: {phone}\n"
        f"Email: {email}\n"
        f"City: {city}\n\n"
        f"Objective\n"
        f"Seeking a challenging position to apply my technical skills.\n\n"
        f"Academic Qualifications\n"
        f"{degree} from {college} ({grad_year})\n\n"
        f"Skills Known\n"
        f"{skills_str}\n\n"
        f"Projects\n"
        f"* E-Commerce Website using React and Node.js\n"
        f"* Student Management System using Python and MySQL\n"
    )

    entities = []
    add_entity(entities, text, name,      "Name")
    add_entity(entities, text, phone,     "Phone")
    add_entity(entities, text, email,     "Email")
    add_entity(entities, text, college,   "College")
    add_entity(entities, text, degree,    "Degree")
    add_entity(entities, text, grad_year, "Graduation_Year")
    for skill in skills:
        add_entity(entities, text, skill, "Skill")

    return text.strip(), entities


def build_v4_senior():
    """Senior engineer — multiple companies, certifications."""
    name        = random_name()
    email       = random_email(name)
    phone       = random.choice(PHONES)
    city        = random.choice(CITIES)
    college     = random.choice(COLLEGES)
    degree      = random.choice(DEGREES)
    grad_year   = random_year(2010, 2017)
    company1    = random.choice(COMPANIES)
    company2    = random.choice([c for c in COMPANIES if c != company1])
    company3    = random.choice([c for c in COMPANIES if c not in [company1, company2]])
    designation = random.choice(DESIGNATIONS)
    skills      = random_skills(random.randint(8, 12))
    skills_str  = " | ".join(skills)
    cert1       = random.choice(CERTIFICATIONS)
    cert2       = random.choice([c for c in CERTIFICATIONS if c != cert1])
    exp_yrs     = str(random.randint(5, 12))

    text = (
        f"{name}\n"
        f"{phone} | {email} | {city}\n\n"
        f"Professional Summary\n"
        f"{designation} with {exp_yrs} years of experience.\n\n"
        f"Core Skills\n"
        f"{skills_str}\n\n"
        f"Professional Experience\n"
        f"{designation} | {company1} | 3 years\n"
        f"Senior Developer | {company2} | 2 years\n"
        f"Junior Developer | {company3} | 1 year\n\n"
        f"Education\n"
        f"{degree} | {college} | {grad_year}\n\n"
        f"Certifications\n"
        f"* {cert1}\n"
        f"* {cert2}\n"
    )

    entities = []
    add_entity(entities, text, name,        "Name")
    add_entity(entities, text, phone,       "Phone")
    add_entity(entities, text, email,       "Email")
    add_entity(entities, text, college,     "College")
    add_entity(entities, text, degree,      "Degree")
    add_entity(entities, text, grad_year,   "Graduation_Year")
    add_entity(entities, text, company1,    "Company")
    add_entity(entities, text, company2,    "Company")
    add_entity(entities, text, company3,    "Company")
    add_entity(entities, text, designation, "Designation")
    add_entity(entities, text, cert1,       "Certification")
    add_entity(entities, text, cert2,       "Certification")
    for skill in skills:
        add_entity(entities, text, skill, "Skill")

    return text.strip(), entities


# ── Main ──────────────────────────────────────────────────────────────────────

def generate_dataset(n=1200):
    builders = [build_v1_standard, build_v2_compact, build_v3_fresher, build_v4_senior]
    dataset  = []
    skipped  = 0

    for i in range(n):
        builder = random.choice(builders)
        try:
            text, entities = builder()
            entities = remove_overlapping(entities)

            if not text.strip() or len(entities) == 0:
                skipped += 1
                continue

            dataset.append({"text": text, "entities": entities})

        except Exception as e:
            skipped += 1
            print(f"  Sample {i} failed: {e}")

    print(f"Generated : {len(dataset)} samples")
    print(f"Skipped   : {skipped} samples")
    return dataset


if __name__ == "__main__":
    print("=" * 50)
    print("Step 1 — Generating NER training data")
    print("=" * 50)

    dataset = generate_dataset(1200)

    with open(NER_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"Saved to  : {NER_DATA_PATH}")
    print(f"Total     : {len(dataset)} samples")

    # Print one sample
    print("\n── Sample preview ────────────────────────────────────")
    s = dataset[0]
    print(s["text"][:250])
    print("\nEntities:")
    for ent in s["entities"][:8]:
        start, end, label = ent
        print(f"  [{label:20s}] {s['text'][start:end]!r}")

    # Label distribution
    print("\n── Entity distribution ───────────────────────────────")
    all_labels = [e[2] for s in dataset for e in s["entities"]]
    for label, count in Counter(all_labels).most_common():
        print(f"  {label:25s} {count}")

    print("\nStep 1 complete! Next: run step2_generate_ranking_dataset.py")