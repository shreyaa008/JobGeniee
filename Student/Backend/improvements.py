ADVANCED     = {'DSA', 'Machine Learning', 'Deep Learning', 'TensorFlow', 'PyTorch'}
INTERMEDIATE = {'Python', 'Java', 'React', 'AWS', 'Docker', 'Node.js', 'Django'}


def get_improvements(cgpa, skills, internships, github_projects,
                     hackathons, backlogs, certifications, aptitude_score):
    """
    Returns a list of personalised improvement suggestions based on
    the student's current profile. Each item is a dict with:
        - priority: 'high' | 'medium' | 'low'
        - message:  the suggestion text
    """
    suggestions = []
    skill_set   = set(skills)
    adv_count   = len(skill_set & ADVANCED)

    # ── High priority ─────────────────────────────────────────────────────────
    if backlogs > 0:
        suggestions.append({
            "priority": "high",
            "message":  f"Clear your {backlogs} active backlog(s) — they directly reduce predicted salary."
        })

    if cgpa < 6.0:
        suggestions.append({
            "priority": "high",
            "message":  "Focus on improving your CGPA above 6.0 — it is the strongest salary predictor."
        })

    if adv_count == 0:
        suggestions.append({
            "priority": "high",
            "message":  "Learn at least one advanced skill (DSA, Machine Learning, Deep Learning, "
                        "TensorFlow, or PyTorch) to unlock higher salary brackets."
        })

    if aptitude_score < 40:
        suggestions.append({
            "priority": "high",
            "message":  f"Your aptitude score is {aptitude_score:.0f}/100. "
                        "Practice quantitative aptitude and logical reasoning regularly."
        })

    # ── Medium priority ───────────────────────────────────────────────────────
    if internships == 0:
        suggestions.append({
            "priority": "medium",
            "message":  "Complete at least 1 internship — practical experience significantly "
                        "improves job prospects and salary offers."
        })

    if github_projects < 2:
        suggestions.append({
            "priority": "medium",
            "message":  f"You have {github_projects} GitHub project(s). "
                        "Build at least 2 end-to-end projects to show practical skills."
        })

    if 6.0 <= cgpa < 7.0:
        suggestions.append({
            "priority": "medium",
            "message":  "Pushing your CGPA above 7.0 will increase your predicted salary by 2–4 LPA."
        })

    if aptitude_score < 60:
        suggestions.append({
            "priority": "medium",
            "message":  f"Aptitude score {aptitude_score:.0f}/100 — aim for 60+ "
                        "to match profiles that typically get higher offers."
        })

    # ── Low priority ──────────────────────────────────────────────────────────
    if hackathons == 0:
        suggestions.append({
            "priority": "low",
            "message":  "Participate in at least 1 hackathon — it demonstrates initiative "
                        "and builds your portfolio."
        })

    if certifications == 0:
        suggestions.append({
            "priority": "low",
            "message":  "Get at least 1 relevant certification "
                        "(AWS, Google, Coursera, etc.) to strengthen your profile."
        })

    if adv_count == 0 and len(skill_set & INTERMEDIATE) < 2:
        suggestions.append({
            "priority": "low",
            "message":  "Broaden your skill set — learn a second intermediate skill "
                        "(Python, Java, React, Docker, AWS) alongside your current ones."
        })

    # ── All good ──────────────────────────────────────────────────────────────
    if not suggestions:
        suggestions.append({
            "priority": "low",
            "message":  "Strong profile! Keep building projects and staying updated "
                        "with industry trends."
        })

    return suggestions