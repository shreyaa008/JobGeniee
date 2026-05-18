"""
utils/parser.py
----------------
Handles PDF text extraction and NER entity parsing.
Simple functions — one job each.
"""

import os
import re
import spacy
import pdfplumber


SKILLS_LIST = [
    "Python","Java","C++","C","JavaScript","TypeScript","Go","Rust",
    "Kotlin","Swift","R","Scala","PHP","React","Angular","Vue.js","Node.js",
    "Django","Flask","FastAPI","Spring Boot","Express.js","Next.js","HTML","CSS",
    "Pandas","NumPy","Scikit-learn","TensorFlow","PyTorch","Keras","XGBoost",
    "Matplotlib","Seaborn","Tableau","Power BI","MySQL","PostgreSQL","MongoDB",
    "Redis","SQLite","Cassandra","Oracle DB","Firebase","AWS","Azure","GCP",
    "Docker","Kubernetes","Jenkins","Git","GitHub Actions","Terraform","Linux",
    "Bash","React Native","Flutter","Android SDK","iOS SDK","REST API","GraphQL",
    "Microservices","CI/CD","Agile","Scrum","System Design","Data Structures","Algorithms",
]

# Valid college keywords — if extracted college contains none of these it is rejected
COLLEGE_KEYWORDS = [
    "university","institute","college","iit","nit","bits","vit","srm",
    "manipal","amity","anna","delhi","mumbai","pune","technology","engineering",
    "school","academy","polytechnic",
]


def load_ner_model(model_path):
    """Load spaCy NER model. Falls back to keyword matching if not found."""
    try:
        nlp = spacy.load(model_path)
        print(f"NER model loaded from {model_path}")
        return nlp, True
    except Exception:
        print("NER model not found — using keyword matching fallback")
        nlp = spacy.blank("en")
        return nlp, False


def extract_text_from_pdf(pdf_path):
    """Extract raw text from a PDF file."""
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"  Error reading PDF {pdf_path}: {e}")
    return text


def extract_experience(text):
    """
    Extract years of experience using strict regex.
    Only matches patterns like '3 years', '5+ yrs of experience'.
    """
    patterns = [
        r"(\d{1,2})\+?\s*years?\s+of\s+(?:IT\s+)?experience",
        r"(\d{1,2})\+?\s*yrs?\s+of\s+(?:IT\s+)?experience",
        r"experience\s+of\s+(\d{1,2})\+?\s*years?",
        r"(\d{1,2})\+?\s*years?\s+experience",
    ]
    text_lower = text.lower()
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            try:
                val = float(match.group(1))
                if 0 <= val <= 50:
                    return val
            except (ValueError, IndexError):
                pass
    return 0.0


def extract_skills_by_keyword(text):
    """Fallback: match skills against SKILLS_LIST using regex."""
    text_lower = text.lower()
    found = []
    for skill in SKILLS_LIST:
        if re.search(r"\b" + re.escape(skill.lower()) + r"\b", text_lower):
            found.append(skill)
    return found


def is_valid_college(text):
    """
    Validate extracted college name.
    Rejects values that look like sentences or tech terms.
    """
    if not text:
        return False
    text_lower = text.lower()
    # Must contain at least one college keyword
    has_keyword = any(kw in text_lower for kw in COLLEGE_KEYWORDS)
    # Must not be too long (sentences are not college names)
    not_too_long = len(text.split()) <= 8
    # Must not contain common tech/work words
    bad_words = ["implemented","developed","worked","using","system","api","ci/cd","pipeline"]
    no_bad_words = not any(bw in text_lower for bw in bad_words)
    return has_keyword and not_too_long and no_bad_words


def parse_resume(pdf_path, nlp, use_ner):
    """
    Parse a single resume PDF.
    Returns a dict with all extracted fields, or None if parsing fails.
    """
    try:
        text = extract_text_from_pdf(pdf_path)
        if not text.strip():
            print(f"  No text found in {pdf_path}")
            return None

        result = {
            "filename"      : os.path.basename(pdf_path),
            "name"          : "",
            "email"         : "",
            "phone"         : "",
            "college"       : "",
            "degree"        : "",
            "grad_year"     : "",
            "companies"     : [],
            "designation"   : "",
            "skills"        : [],
            "experience_yrs": 0.0,
        }

        if use_ner:
            doc = nlp(text)
            for ent in doc.ents:
                label = ent.label_
                value = ent.text.strip()
                if not value:
                    continue
                if label == "Name" and not result["name"]:
                    result["name"] = value
                elif label == "Email" and not result["email"]:
                    result["email"] = value
                elif label == "Phone" and not result["phone"]:
                    result["phone"] = value
                elif label == "College" and not result["college"]:
                    # Validate before accepting
                    if is_valid_college(value):
                        result["college"] = value
                elif label == "Degree" and not result["degree"]:
                    result["degree"] = value
                elif label == "Graduation_Year" and not result["grad_year"]:
                    result["grad_year"] = value
                elif label == "Company":
                    if value not in result["companies"]:
                        result["companies"].append(value)
                elif label == "Designation" and not result["designation"]:
                    result["designation"] = value
                elif label == "Skill":
                    if value not in result["skills"]:
                        result["skills"].append(value)

        # Always use keyword matching as skill backup
        if not result["skills"]:
            result["skills"] = extract_skills_by_keyword(text)

        # Always use regex for experience (more reliable)
        result["experience_yrs"] = extract_experience(text)

        # Fallback: name from first line
        if not result["name"]:
            first_line = text.strip().split("\n")[0].strip()
            if first_line and len(first_line.split()) <= 4:
                result["name"] = first_line

        # Fallback: regex for email
        if not result["email"]:
            match = re.search(r"[\w.\-]+@[\w.\-]+\.\w+", text)
            if match:
                result["email"] = match.group()

        # Fallback: regex for phone
        if not result["phone"]:
            match = re.search(r"\b[6-9]\d{9}\b", text)
            if match:
                result["phone"] = match.group()

        return result

    except Exception as e:
        print(f"  Failed to parse {pdf_path}: {e}")
        return None