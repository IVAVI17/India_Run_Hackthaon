# Redrob Intelligent Candidate Ranker

## Architecture

A **hybrid multi-signal scoring system** that ranks candidates the way a senior recruiter would — by understanding role fit, skill depth, career trajectory, and behavioral availability.

### Scoring Components

| Component | Weight | What it measures |
|-----------|--------|-----------------|
| Title / Role Fit | 30% | Is the candidate's career in AI/ML/IR engineering? |
| Skill Depth | 28% | Production IR, embedding, vector DB, ranking experience |
| Experience Arc | 18% | YoE in sweet spot (5-9yr), product co. vs. services ratio |
| Career Narrative | 12% | NLP scan of job descriptions for production AI signals |
| Location | 7% | India tier-1 city preference (Pune/Noida/Hyderabad/Bangalore) |
| Education | 5% | Institution tier + relevant field |
| Behavioral Multiplier | ×0.5–1.3 | Engagement, recency, notice period, GitHub activity |

### Key Design Decisions

1. **Title is the decisive anti-keyword-stuffer signal.** A Marketing Manager with all the AI skill labels gets a 0.05 title score — killed before skills are even considered.

2. **Skill scoring requires duration + proficiency + endorsements.** Expert skills with 0 months duration are penalized (honeypot/keyword-stuffer signal).

3. **Behavioral signals are a *multiplier*, not additive.** A great-on-paper candidate who hasn't logged in for 6 months gets multiplied down. A slightly weaker candidate who is actively engaged gets a boost.

4. **Honeypot detection** catches impossible profiles (YoE >> career history, expert skills with 0 duration, impossible date ranges).

5. **Services-company career penalty.** If >85% of career is at TCS/Infosys/Wipro etc., score is penalized per JD guidance.

## How to Run

```bash
# Install (no dependencies needed for core ranker)
pip install pandas  # optional, for exploration only

# Run on full dataset
python rank.py --candidates candidates.jsonl --out submission.csv

# Or gzipped
python rank.py --candidates candidates.jsonl.gz --out submission.csv

# Validate
python validate_submission.py submission.csv
```

**Runtime:** ~60–90 seconds for 100K candidates on a single CPU core.  
**Memory:** <1 GB RAM.  
**No network, no GPU required.**

## Files

```
rank.py                  — Main ranker (run this)
requirements.txt         — Dependencies (stdlib only for ranker)
README.md                — This file
submission_metadata.yaml — Fill in your team details before submitting
```
