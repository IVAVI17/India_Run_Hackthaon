"""
Redrob Candidate Ranker — Streamlit Sandbox
============================================
Deploy this on HuggingFace Spaces or Streamlit Cloud.
Upload a small candidates JSON/JSONL file and get the ranked output.
Required for hackathon submission (Section 10.5 of submission_spec).
"""

import json
import csv
import io
import sys
import streamlit as st
import pandas as pd
from pathlib import Path

# Import the ranker
sys.path.insert(0, str(Path(__file__).parent))
from rank import score_candidate

st.set_page_config(page_title="Redrob Candidate Ranker", page_icon="🎯", layout="wide")

st.title("🎯 Redrob Intelligent Candidate Ranker")
st.caption("Hybrid multi-signal ranking engine for the Redrob Hackathon")

with st.expander("ℹ️ How this works", expanded=False):
    st.markdown("""
    **Architecture: Hybrid Multi-Signal Scorer + Behavioral Multiplier**
    
    | Component | Weight | Signal |
    |-----------|--------|--------|
    | Title / Role Fit | 30% | Is this an AI/ML/IR engineering role? |
    | Skill Depth | 28% | Production embedding, vector DB, ranking experience |
    | Experience Arc | 18% | 5-9yr sweet spot, product co. vs services ratio |
    | Career Narrative | 12% | NLP scan for production AI signals in job descriptions |
    | Location | 7% | India tier-1 city preference |
    | Education | 5% | Institution tier + relevant field |
    | **Behavioral Multiplier** | **×0.5–1.3** | Engagement, recency, notice period |
    
    **Key anti-patterns detected:**
    - Title mismatch (Marketing Manager with AI skills → rejected)
    - Keyword stuffing (expert skills with 0 months duration → penalized)
    - Pure services careers (>85% at TCS/Infosys/etc. → penalized)
    - Inactive candidates (6+ months since login → multiplier down)
    - Honeypot profiles (impossible date/experience data → score = 0)
    """)

st.markdown("---")
st.subheader("Upload Candidates")
st.info("Upload a JSON array or JSONL file of candidates (the sample_candidates.json from the bundle works perfectly).")

uploaded = st.file_uploader("Choose a candidates file", type=["json", "jsonl"])

if uploaded:
    try:
        raw = uploaded.read().decode("utf-8")
        # Try JSON array first, then JSONL
        try:
            candidates = json.loads(raw)
            if not isinstance(candidates, list):
                candidates = [candidates]
        except json.JSONDecodeError:
            candidates = [json.loads(line) for line in raw.splitlines() if line.strip()]

        st.success(f"Loaded **{len(candidates)}** candidates")

        if st.button("🚀 Run Ranker", type="primary"):
            with st.spinner("Scoring candidates..."):
                scored = []
                prog = st.progress(0)
                for i, c in enumerate(candidates):
                    score, reason = score_candidate(c)
                    scored.append({
                        "rank": None,
                        "candidate_id": c["candidate_id"],
                        "score": round(score, 6),
                        "current_title": c["profile"]["current_title"],
                        "company": c["profile"]["current_company"],
                        "years_exp": c["profile"]["years_of_experience"],
                        "location": f"{c['profile']['location']}, {c['profile']['country']}",
                        "reasoning": reason,
                    })
                    prog.progress((i + 1) / len(candidates))

            # Sort and assign ranks
            scored.sort(key=lambda x: (-x["score"], x["candidate_id"]))
            for i, row in enumerate(scored):
                row["rank"] = i + 1

            df = pd.DataFrame(scored)

            st.markdown("---")
            st.subheader(f"🏆 Ranked Results (Top {min(100, len(scored))})")

            # Score distribution
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Candidates Scored", len(scored))
            col2.metric("Top Score", f"{scored[0]['score']:.3f}")
            col3.metric("Median Score", f"{df['score'].median():.3f}")
            col4.metric("Candidates in Top 100", min(100, len(scored)))

            # Show table
            display_cols = ["rank", "candidate_id", "score", "current_title", "company", "years_exp", "location"]
            st.dataframe(
                df[display_cols].head(100),
                use_container_width=True,
                hide_index=True,
            )

            # Reasoning expander
            with st.expander("📝 View Reasonings (Top 20)"):
                for row in scored[:20]:
                    st.markdown(f"**#{row['rank']} {row['candidate_id']}** ({row['score']:.3f})")
                    st.caption(row["reasoning"])
                    st.divider()

            # Download CSV
            csv_rows = [["candidate_id", "rank", "score", "reasoning"]]
            for row in scored[:100]:
                csv_rows.append([row["candidate_id"], row["rank"], row["score"], row["reasoning"]])

            buf = io.StringIO()
            writer = csv.writer(buf)
            writer.writerows(csv_rows)

            st.download_button(
                label="⬇️ Download submission.csv",
                data=buf.getvalue().encode("utf-8"),
                file_name="submission.csv",
                mime="text/csv",
            )

    except Exception as e:
        st.error(f"Error processing file: {e}")
        st.exception(e)

else:
    st.markdown("""
    **Quick start:** Upload the `sample_candidates.json` from the hackathon bundle to see the ranker in action.
    
    For the full submission, run locally:
    ```bash
    python rank.py --candidates candidates.jsonl.gz --out submission.csv
    python validate_submission.py submission.csv
    ```
    """)
