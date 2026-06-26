#!/usr/bin/env python3
"""
Redrob Intelligent Candidate Discovery & Ranking Engine
========================================================
Author: Team Submission
Architecture: Hybrid Multi-Signal Scorer with Behavioral Multiplier

Ranking philosophy:
  1. Title/Role fit  — is their career trajectory relevant to AI/ML engineering?
  2. Skill depth     — do they have production IR/ranking/embedding experience?
  3. Experience arc  — right tenure at product companies, not pure services?
  4. Location        — India-based preference (Pune/Noida/NCR/Hyderabad/Bangalore)
  5. Behavioral mux  — engagement signals as a multiplier, not a primary score
  6. Anti-patterns   — explicit down-weighting for disqualifier signals in the JD

Compute budget: < 5 min, < 16 GB RAM, CPU only, no network.
"""

import argparse
import csv
import gzip
import json
import math
import re
import sys
from datetime import date, datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants / configuration
# ---------------------------------------------------------------------------

TODAY = date(2026, 6, 16)   # Competition reference date

# AI/ML/IR core skills that directly matter for the JD
CORE_SKILLS = {
    # Embeddings & retrieval — MUST HAVE category
    "sentence-transformers", "sentence transformers", "embeddings", "vector embeddings",
    "dense retrieval", "bi-encoder", "cross-encoder", "neural search",
    # Vector DBs
    "pinecone", "weaviate", "qdrant", "milvus", "faiss", "opensearch",
    "elasticsearch", "chroma", "pgvector", "annoy",
    # Retrieval / ranking systems
    "information retrieval", "bm25", "hybrid search", "lexical search",
    "learning to rank", "ltr", "ranking", "re-ranking", "reranking",
    "recommendation system", "recommender system", "recommendation engine",
    # LLMs
    "llm", "large language model", "gpt", "llama", "mistral", "gemma",
    "fine-tuning", "fine tuning", "lora", "qlora", "peft",
    "rag", "retrieval augmented generation", "retrieval-augmented generation",
    # NLP
    "nlp", "natural language processing", "text classification", "named entity recognition",
    "ner", "transformers", "bert", "roberta", "t5", "hugging face", "huggingface",
    # Evaluation
    "ndcg", "mrr", "map", "mean average precision", "a/b testing", "ab testing",
    "evaluation framework", "offline evaluation", "online evaluation",
    # Python & ML fundamentals
    "pytorch", "tensorflow", "scikit-learn", "sklearn", "xgboost", "lightgbm",
    "mlflow", "wandb", "weights & biases",
}

# Good-signal skills (secondary, not core)
SECONDARY_SKILLS = {
    "python", "pyspark", "spark", "kafka", "airflow",
    "docker", "kubernetes", "redis", "postgresql", "sql",
    "aws", "gcp", "azure", "cloud", "mlops", "data engineering",
    "distributed systems", "system design", "api design", "rest api",
    "git", "github", "open source",
}

# Anti-pattern skills/titles that indicate non-fit
ANTI_TITLES = {
    "marketing manager", "hr manager", "accountant", "content writer",
    "customer support", "operations manager", "business analyst", "sales",
    "project manager", "finance", "graphic designer", "ui/ux designer",
    "seo specialist", "recruiter", "hr business partner",
}

# Pure services companies (JD explicitly flags as disqualifier if entire career)
SERVICES_COMPANIES = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "tech mahindra", "mphasis", "hexaware", "l&t infotech",
    "mindtree",  # acquired by L&T, services-oriented
}

# Preferred India locations
INDIA_TIER1_LOCATIONS = {
    "noida", "pune", "gurgaon", "gurugram", "delhi", "new delhi",
    "bangalore", "bengaluru", "hyderabad", "mumbai", "navi mumbai",
    "chennai", "ncr",
}

# ---------------------------------------------------------------------------
# Honeypot detection helpers
# ---------------------------------------------------------------------------

def is_likely_honeypot(candidate: dict) -> bool:
    """
    Detect profiles with logically impossible data.
    - Experience at a company longer than the company has existed
    - 'expert' in 10+ skills all with 0 months usage
    - Years of experience < sum of career history (big gap)
    """
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])

    # Check 1: expert skills with zero duration (keyword stuffing indicator)
    expert_zero_duration = sum(
        1 for s in skills
        if s.get("proficiency") == "expert" and s.get("duration_months", 1) == 0
    )
    if expert_zero_duration >= 5:
        return True

    # Check 2: duration months inconsistency in career
    for role in career:
        start_str = role.get("start_date", "")
        end_str = role.get("end_date") or str(TODAY)
        try:
            start = datetime.strptime(start_str[:7], "%Y-%m").date()
            end = datetime.strptime(end_str[:7], "%Y-%m").date()
            actual_months = (end.year - start.year) * 12 + (end.month - start.month)
            stated_months = role.get("duration_months", 0)
            # If stated duration is 2x actual or impossible (negative actual)
            if actual_months < 0 or (stated_months > 0 and stated_months > actual_months * 2.5):
                return True
        except (ValueError, TypeError):
            pass

    # Check 3: YoE much larger than career history allows
    total_career_months = sum(r.get("duration_months", 0) for r in career)
    yoe_months = profile.get("years_of_experience", 0) * 12
    # If claimed YoE is more than 5 years beyond what career records show
    if yoe_months > total_career_months + 60 and total_career_months > 0:
        return True

    return False


# ---------------------------------------------------------------------------
# Scoring components
# ---------------------------------------------------------------------------

def score_title_role(candidate: dict) -> float:
    """
    0.0–1.0. Is the candidate's title/career in the right area?
    This is the strongest anti-keyword-stuffer signal.
    """
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])

    current_title = profile.get("current_title", "").lower()
    headline = profile.get("headline", "").lower()
    summary = profile.get("summary", "").lower()

    # Hard negative: anti-pattern titles
    for bad in ANTI_TITLES:
        if bad in current_title:
            return 0.05   # almost zero — not a match

    # Positive title signals
    ai_ml_titles = [
        "ml engineer", "machine learning engineer", "ai engineer", "nlp engineer",
        "data scientist", "research engineer", "applied scientist",
        "search engineer", "ranking engineer", "recommendation engineer",
        "backend engineer", "software engineer", "senior engineer",
        "staff engineer", "principal engineer", "tech lead",
    ]
    for good in ai_ml_titles:
        if good in current_title:
            base = 0.7
            # Bonus for explicit ML/AI/IR in title
            if any(w in current_title for w in ["ml", "ai", "nlp", "search", "ranking", "data science"]):
                base = 0.95
            return base

    # If title is vague, look at career trajectory
    ml_roles_in_history = sum(
        1 for r in career
        if any(kw in r.get("title", "").lower()
               for kw in ["ml", "ai", "nlp", "data scient", "research", "ranking", "recommendation"])
    )
    if ml_roles_in_history >= 2:
        return 0.75
    if ml_roles_in_history == 1:
        return 0.5

    return 0.2


def score_skills(candidate: dict) -> float:
    """
    0.0–1.0. Measures depth of relevant skills.
    Considers: skill name, proficiency level, endorsements, and duration.
    Weighted toward core skills with evidence of actual use.
    """
    skills = candidate.get("skills", [])
    redrob = candidate.get("redrob_signals", {})
    assessment_scores = redrob.get("skill_assessment_scores", {})

    core_score = 0.0
    secondary_score = 0.0

    for skill in skills:
        name_lower = skill.get("name", "").lower()
        proficiency = skill.get("proficiency", "beginner")
        endorsements = min(skill.get("endorsements", 0), 50)   # cap outlier boosters
        duration = skill.get("duration_months", 0)

        prof_weights = {"beginner": 0.2, "intermediate": 0.5, "advanced": 0.8, "expert": 1.0}
        prof_w = prof_weights.get(proficiency, 0.2)

        # Duration multiplier: more months = more credibility
        dur_mult = min(1.0, 0.3 + (duration / 60.0) * 0.7)  # 0 months → 0.3, 60+ months → 1.0
        if duration == 0:
            dur_mult = 0.25  # suspicious zero-duration expert skills

        # Endorsement bonus (log scale)
        endorse_bonus = math.log1p(endorsements) / math.log1p(50) * 0.2

        is_core = any(kw in name_lower for kw in CORE_SKILLS)
        is_secondary = any(kw in name_lower for kw in SECONDARY_SKILLS)

        # Check if there's a platform assessment for this skill
        assess_bonus = 0.0
        for assess_name, assess_score in assessment_scores.items():
            if assess_name.lower() in name_lower or name_lower in assess_name.lower():
                assess_bonus = (assess_score / 100.0) * 0.15
                break

        skill_val = prof_w * dur_mult + endorse_bonus + assess_bonus

        if is_core:
            core_score += skill_val
        elif is_secondary:
            secondary_score += skill_val * 0.4

    # Normalize: a strong candidate might have 5-8 core skills at full weight
    core_score = min(1.0, core_score / 4.0)
    secondary_score = min(0.3, secondary_score / 4.0)

    return min(1.0, core_score * 0.8 + secondary_score * 0.2)


def score_experience(candidate: dict) -> float:
    """
    0.0–1.0. Years of experience and career arc quality.
    Sweet spot: 5-9 years at product companies, not pure services.
    """
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])

    yoe = profile.get("years_of_experience", 0)

    # YoE scoring: sweet spot 5-9, penalties outside
    if 5 <= yoe <= 9:
        yoe_score = 1.0
    elif 4 <= yoe < 5 or 9 < yoe <= 12:
        yoe_score = 0.8
    elif 3 <= yoe < 4 or 12 < yoe <= 15:
        yoe_score = 0.5
    elif yoe < 3:
        yoe_score = 0.2
    else:
        yoe_score = 0.4   # >15 years — possible overexperience for founding team

    # Career arc: product vs services
    total_months = sum(r.get("duration_months", 0) for r in career)
    services_months = sum(
        r.get("duration_months", 0) for r in career
        if any(co in r.get("company", "").lower() for co in SERVICES_COMPANIES)
    )

    if total_months > 0:
        services_ratio = services_months / total_months
    else:
        services_ratio = 0.0

    # Pure services career → big penalty (JD explicitly flags this)
    if services_ratio >= 0.85:
        career_penalty = 0.4
    elif services_ratio >= 0.5:
        career_penalty = 0.7
    else:
        career_penalty = 1.0

    # Tenure signal: did they stay long enough? (anti-title-chaser check)
    # Average tenure >= 24 months is a positive signal
    if len(career) > 0:
        avg_tenure = total_months / len(career)
        tenure_score = min(1.0, avg_tenure / 30.0)   # 30 months = full score
    else:
        tenure_score = 0.5

    return yoe_score * 0.5 * career_penalty + tenure_score * 0.3 * career_penalty + (1 - services_ratio) * 0.2


def score_location(candidate: dict) -> float:
    """
    0.0–1.0. India-based, right cities preferred.
    """
    profile = candidate.get("profile", {})
    country = profile.get("country", "").lower()
    location = profile.get("location", "").lower()
    redrob = candidate.get("redrob_signals", {})
    willing_to_relocate = redrob.get("willing_to_relocate", False)

    if country == "india":
        for tier1 in INDIA_TIER1_LOCATIONS:
            if tier1 in location:
                return 1.0
        # India but not tier 1 city — still good
        return 0.7
    else:
        # Outside India — possible but lower priority unless willing to relocate
        return 0.3 if willing_to_relocate else 0.1


def score_education(candidate: dict) -> float:
    """
    0.0–1.0. Education quality (lighter weight than skills/experience).
    """
    education = candidate.get("education", [])

    if not education:
        return 0.3   # no education info — neutral-negative

    best_score = 0.0
    for edu in education:
        tier = edu.get("tier", "unknown")
        field = edu.get("field_of_study", "").lower()
        degree = edu.get("degree", "").lower()

        tier_scores = {
            "tier_1": 1.0,
            "tier_2": 0.8,
            "tier_3": 0.6,
            "tier_4": 0.4,
            "unknown": 0.35,
        }
        t_score = tier_scores.get(tier, 0.35)

        # Field bonus for CS, AI, ML, statistics
        field_bonus = 0.0
        if any(kw in field for kw in ["computer science", "ai", "ml", "machine learning",
                                        "data science", "statistics", "information"]):
            field_bonus = 0.1

        # Degree level bonus
        if "phd" in degree or "doctorate" in degree:
            deg_bonus = 0.1
        elif "master" in degree or "m.tech" in degree or "ms " in degree:
            deg_bonus = 0.05
        else:
            deg_bonus = 0.0

        score = min(1.0, t_score + field_bonus + deg_bonus)
        best_score = max(best_score, score)

    return best_score


def score_behavioral(candidate: dict) -> float:
    """
    0.0–1.5 (used as a multiplier, can slightly boost or penalize).
    Measures engagement, availability, and platform activity.
    """
    sig = candidate.get("redrob_signals", {})

    # --- Availability signals ---
    open_to_work = sig.get("open_to_work_flag", False)
    notice_days = sig.get("notice_period_days", 90)

    avail_score = 0.5
    if open_to_work:
        avail_score += 0.3
    if notice_days <= 30:
        avail_score += 0.2
    elif notice_days <= 60:
        avail_score += 0.1
    # 90+ days notice → no bonus, slight negative
    elif notice_days > 90:
        avail_score -= 0.1

    # --- Recency of activity ---
    last_active_str = sig.get("last_active_date", "")
    try:
        last_active = datetime.strptime(last_active_str, "%Y-%m-%d").date()
        days_inactive = (TODAY - last_active).days
        if days_inactive <= 30:
            activity_score = 1.0
        elif days_inactive <= 90:
            activity_score = 0.7
        elif days_inactive <= 180:
            activity_score = 0.4
        else:
            activity_score = 0.1   # hasn't logged in for 6+ months
    except (ValueError, TypeError):
        activity_score = 0.5

    # --- Engagement quality ---
    response_rate = sig.get("recruiter_response_rate", 0.5)
    response_time = sig.get("avg_response_time_hours", 48)
    interview_rate = sig.get("interview_completion_rate", 0.5)

    engagement_score = (
        response_rate * 0.4 +
        max(0, (1.0 - min(response_time, 168) / 168)) * 0.3 +
        interview_rate * 0.3
    )

    # --- Platform signals ---
    profile_completeness = sig.get("profile_completeness_score", 50) / 100.0
    github_score = sig.get("github_activity_score", -1)
    github_norm = (github_score / 100.0) if github_score >= 0 else 0.3

    saved_30d = min(sig.get("saved_by_recruiters_30d", 0), 20)
    saved_norm = saved_30d / 20.0

    platform_score = (profile_completeness * 0.4 + github_norm * 0.35 + saved_norm * 0.25)

    # --- Combine into multiplier ---
    raw = (avail_score * 0.35 + activity_score * 0.30 + engagement_score * 0.20 + platform_score * 0.15)

    # Map to [0.5, 1.3] so even a poor behavioral candidate isn't zeroed out
    multiplier = 0.5 + raw * 0.8
    return multiplier


def score_career_narrative(candidate: dict) -> float:
    """
    0.0–1.0. Does the career history text show production AI/ML/IR work?
    Simple keyword scan over descriptions — catches Tier 5 candidates who
    don't have AI skill labels but built relevant systems.
    """
    career = candidate.get("career_history", [])
    profile = candidate.get("profile", {})
    summary_text = profile.get("summary", "").lower()
    headline_text = profile.get("headline", "").lower()

    all_text = summary_text + " " + headline_text
    for role in career:
        all_text += " " + role.get("description", "").lower()

    # Key production signals in text
    production_signals = [
        "embedding", "vector", "retrieval", "ranking", "recommendation",
        "nlp", "bert", "llm", "fine-tun", "rag",
        "search engine", "semantic search", "dense retrieval",
        "recommendation system", "ranking system", "ranking model",
        "ndcg", "a/b test", "evaluation framework",
        "shipped", "deployed", "production", "real users", "at scale",
        "pinecone", "weaviate", "faiss", "opensearch", "elasticsearch",
        "sentence-transformer", "hugging face", "transformers",
    ]

    matches = sum(1 for sig in production_signals if sig in all_text)
    return min(1.0, matches / 6.0)


# ---------------------------------------------------------------------------
# Master scoring function
# ---------------------------------------------------------------------------

def score_candidate(candidate: dict) -> tuple[float, str]:
    """
    Returns (composite_score, reasoning_string).
    """
    if is_likely_honeypot(candidate):
        return 0.0, "Profile contains inconsistent data signals (likely synthetic/invalid)."

    # Component scores
    title_score   = score_title_role(candidate)
    skill_score   = score_skills(candidate)
    exp_score     = score_experience(candidate)
    loc_score     = score_location(candidate)
    edu_score     = score_education(candidate)
    narr_score    = score_career_narrative(candidate)
    behav_mult    = score_behavioral(candidate)

    # Hard disqualifier: if title score is near zero, can't be in top candidates
    if title_score <= 0.05:
        return title_score * 0.1, "Current role is outside AI/ML/engineering domain."

    # Weighted combination (weights sum to 1.0)
    # Title/role fit is decisive against keyword stuffers
    base_score = (
        title_score  * 0.30 +
        skill_score  * 0.28 +
        exp_score    * 0.18 +
        narr_score   * 0.12 +
        loc_score    * 0.07 +
        edu_score    * 0.05
    )

    # Apply behavioral multiplier
    final_score = base_score * behav_mult

    # Clamp to [0, 1]
    final_score = min(1.0, max(0.0, final_score))

    # Build reasoning
    reasoning = build_reasoning(candidate, title_score, skill_score, exp_score,
                                 loc_score, narr_score, behav_mult)

    return final_score, reasoning


def build_reasoning(candidate, title_score, skill_score, exp_score,
                    loc_score, narr_score, behav_mult):
    """Generate specific, honest 1-2 sentence reasoning."""
    profile = candidate.get("profile", {})
    sig = candidate.get("redrob_signals", {})
    skills = candidate.get("skills", [])

    yoe = profile.get("years_of_experience", 0)
    title = profile.get("current_title", "")
    company = profile.get("current_company", "")
    location = profile.get("location", "")

    # Core skills found
    core_found = [s["name"] for s in skills
                  if any(kw in s["name"].lower() for kw in CORE_SKILLS)][:3]

    notice = sig.get("notice_period_days", 90)
    open_work = sig.get("open_to_work_flag", False)
    last_active_str = sig.get("last_active_date", "")

    try:
        last_active = datetime.strptime(last_active_str, "%Y-%m-%d").date()
        days_inactive = (TODAY - last_active).days
        active_str = f"active {days_inactive}d ago" if days_inactive < 60 else f"inactive {days_inactive}d"
    except Exception:
        active_str = "activity unknown"

    parts = []

    if title_score >= 0.7:
        parts.append(f"{yoe:.0f}yr {title} at {company}")
    else:
        parts.append(f"{yoe:.0f}yr exp, title '{title}' (moderate JD alignment)")

    if core_found:
        parts.append(f"relevant skills: {', '.join(core_found)}")
    elif narr_score >= 0.5:
        parts.append("career descriptions show production AI/IR system experience")
    else:
        parts.append("limited core AI/IR skill signals")

    concern = []
    if notice > 60:
        concern.append(f"{notice}d notice period")
    if not open_work:
        concern.append("not flagged open-to-work")
    if behav_mult < 0.7:
        concern.append(f"{active_str}")
    if loc_score < 0.5:
        concern.append(f"location ({location}) outside India preference")

    if concern:
        parts.append("Concerns: " + "; ".join(concern))

    sentence1 = "; ".join(parts[:2]) + "."
    sentence2 = parts[2] + "." if len(parts) > 2 else ""
    return (sentence1 + " " + sentence2).strip()


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_candidates(path: str):
    """Load candidates from .jsonl or .jsonl.gz file."""
    p = Path(path)
    opener = gzip.open if p.suffix == ".gz" else open
    mode = "rt"
    with opener(p, mode, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_submission(ranked: list[dict], out_path: str):
    """Write the top-100 ranked candidates to CSV."""
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for i, item in enumerate(ranked[:100], start=1):
            writer.writerow([
                item["candidate_id"],
                i,
                f"{item['score']:.6f}",
                item["reasoning"],
            ])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Redrob Candidate Ranker")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl or candidates.jsonl.gz")
    parser.add_argument("--out", default="submission.csv", help="Output CSV path")
    parser.add_argument("--top-n", type=int, default=100, help="Number of candidates to output")
    args = parser.parse_args()

    print(f"Loading candidates from {args.candidates}...", file=sys.stderr)
    scored = []
    for i, candidate in enumerate(load_candidates(args.candidates)):
        score, reasoning = score_candidate(candidate)
        scored.append({
            "candidate_id": candidate["candidate_id"],
            "score": score,
            "reasoning": reasoning,
        })
        if (i + 1) % 10000 == 0:
            print(f"  Processed {i+1} candidates...", file=sys.stderr)

    print(f"Scoring complete. Total candidates: {len(scored)}", file=sys.stderr)

    # Sort by score descending; tie-break by candidate_id ascending (per spec)
    scored.sort(key=lambda x: (-x["score"], x["candidate_id"]))

    # Clamp scores to be monotonically non-increasing (handle float precision)
    for i in range(1, len(scored)):
        if scored[i]["score"] > scored[i - 1]["score"]:
            scored[i]["score"] = scored[i - 1]["score"]

    write_submission(scored, args.out)
    print(f"Written top-{args.top_n} to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
