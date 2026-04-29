from __future__ import annotations

import io
import re
import numpy as np
import streamlit as st
import pypdf
import unicodedata
from sentence_transformers import SentenceTransformer, util

# ── CONFIG ─────────────────────────────────────────────────────
EMBED_MODEL = "all-MiniLM-L6-v2"
PLUS_THR = 0.35
GAP_THR  = 0.18

SKILL_LINKS = {
    "python": (
        "https://www.coursera.org/search?query=python",
        "https://www.udemy.com/courses/search/?q=python"
    ),
    "machine learning": (
        "https://www.coursera.org/search?query=machine%20learning",
        "https://www.udemy.com/courses/search/?q=machine%20learning"
    ),
    "deep learning": (
        "https://www.coursera.org/search?query=deep%20learning",
        "https://www.udemy.com/courses/search/?q=deep%20learning"
    ),
    "c++": (
        "https://www.coursera.org/search?query=c%2B%2B",
        "https://www.udemy.com/courses/search/?q=c%2B%2B"
    ),
    "ros": (
        "https://www.coursera.org/search?query=robot%20operating%20system",
        "https://www.udemy.com/courses/search/?q=ros"
    ),
    "embedded systems": (
        "https://www.coursera.org/search?query=embedded%20systems",
        "https://www.udemy.com/courses/search/?q=embedded%20systems"
    ),
    "computer vision": (
        "https://www.coursera.org/search?query=computer%20vision",
        "https://www.udemy.com/courses/search/?q=computer%20vision"
    ),
}

# ── MODEL ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    return SentenceTransformer(EMBED_MODEL,device="cpu")

# ── TEXT CLEANING ──────────────────────────────────────────────
def extract_pdf_text(uploaded_file) -> str:
    reader = pypdf.PdfReader(io.BytesIO(uploaded_file.read()))
    pages  = []

    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)

    raw = "\n".join(pages)

    # Normalize unicode
    raw = unicodedata.normalize("NFKC", raw)
    raw = raw.replace("\xa0", " ")

    raw = re.sub(r' {2,}', ' ', raw)
    raw = re.sub(r'\n{2,}', '\n', raw)

    return raw.strip()

# ── CONTACT FILTER ─────────────────────────────────────────────
CONTACT_PAT = re.compile(
    r"(@|\blinkedin\b|\bgithub\b|^\+?\d[\d\s\-\(\)]{6,}$"
    r"|\bphone\b|\bemail\b|\baddress\b)",
    re.I
)

# ── CLEAN LINE ─────────────────────────────────────────────────
def _clean_line(line: str) -> str:
    line = re.sub(r"[*_`~#]", "", line)
    return line.strip()

# ── RESUME CHUNKING ────────────────────────────────────────────
def chunk_resume(text: str) -> list:
    raw_lines = text.splitlines()
    expanded = []

    for raw in raw_lines:
        parts = re.split(r'(?<=[.!?])\s+', raw)
        expanded.extend(parts)

    chunks = []
    buffer = ""

    for raw in expanded:
        line = _clean_line(raw)

        if not line:
            continue

        if CONTACT_PAT.search(line) and len(line.split()) < 8:
            continue

        if len(line.split()) < 5:
            buffer = (buffer + " " + line).strip()
        else:
            if buffer:
                chunks.append(buffer)
                buffer = ""
            chunks.append(line)

    if buffer:
        chunks.append(buffer)

    chunks = [c for c in chunks if len(c.split()) >= 4]

    if len(chunks) < 4:
        chunks = re.split(r'(?<=[.!?,;:])\s+|\n+', text)
        chunks = [
            _clean_line(c) for c in chunks
            if len(c.split()) >= 4 and not CONTACT_PAT.search(c)
        ]

    # Sliding window
    if len(chunks) >= 3:
        windowed = []
        for i, c in enumerate(chunks):
            prev = chunks[i - 1] if i > 0 else ""
            nxt  = chunks[i + 1] if i < len(chunks) - 1 else ""
            windowed.append(f"{prev} {c} {nxt}".strip())
        chunks = windowed

    return chunks

# ── JD SPLIT ───────────────────────────────────────────────────
def split_jd(text: str):
    lines = text.splitlines()
    reqs = []

    for line in lines:
        line = _clean_line(line)
        if len(line.split()) >= 5:
            reqs.append(line)

    return reqs

# ── SKILL EXTRACTION (FIXED) ───────────────────────────────────
def extract_skills(text: str):
    text = text.lower()
    found = set()

    for skill in SKILL_LINKS.keys():
        if re.search(rf"\b{re.escape(skill)}\b", text):
            found.add(skill)

    return found

# ── SCORE STRETCH ──────────────────────────────────────────────
def stretch_score(x):
    import math
    return 1 / (1 + math.exp(-(x - 0.30) * 12))

# ── ANALYSIS ───────────────────────────────────────────────────
def run_analysis(model, resume_text, jd_text):
    jd_reqs = split_jd(jd_text)
    res_chunks = chunk_resume(resume_text)

    req_embs   = model.encode(jd_reqs, convert_to_tensor=True)
    chunk_embs = model.encode(res_chunks, convert_to_tensor=True)

    cos_mat = util.cos_sim(req_embs, chunk_embs).cpu().numpy()

    results = []

    for i, req in enumerate(jd_reqs):
        sims = cos_mat[i]
        best_idx = int(np.argmax(sims))
        score = float(sims[best_idx])

        results.append({
            "req": req,
            "score": score,
            "match": res_chunks[best_idx]
        })

    raw_mean = np.mean([r["score"] for r in results])
    final_score = round(stretch_score(raw_mean) * 100, 1)

    return final_score, results, res_chunks

# ── UI ─────────────────────────────────────────────────────────
st.title("📊 InsightRank — Resume Auditor")

resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
jd_text = st.text_area("Paste Job Description")

if st.button("Run Analysis"):
    if not resume_file or not jd_text:
        st.error("Upload resume and paste JD")
    else:
        model = load_model()
        resume_text = extract_pdf_text(resume_file)

        score, results, chunks = run_analysis(model, resume_text, jd_text)

        # Skill analysis
        jd_skills = extract_skills(jd_text)
        resume_skills = extract_skills(resume_text)

        matched_skills = jd_skills & resume_skills
        missing_skills = jd_skills - resume_skills

        st.subheader(f"Overall Score: {score}%")

        # ── TOP MATCHES ──
        st.subheader("Top Matches")
        for r in sorted(results, key=lambda x: x["score"], reverse=True)[:5]:
            st.write(f"**{round(r['score']*100)}%** → {r['req']}")
            st.caption(r["match"])

        # ── SKILLS ──
        st.subheader("✅ Matched Skills")
        if matched_skills:
            st.write(", ".join(sorted(matched_skills)))
        else:
            st.write("No strong skill matches found")

        st.subheader("❌ Missing Skills")
        if missing_skills:
            st.write(", ".join(sorted(missing_skills)))
        else:
            st.write("No major skill gaps detected")

        # ── LEARNING PATH ──
        if missing_skills:
            st.subheader("📚 Learning Path (Recommended Courses)")

            for skill in sorted(missing_skills):
                if skill in SKILL_LINKS:
                    coursera, udemy = SKILL_LINKS[skill]

                    st.markdown(f"### {skill.upper()}")
                    st.markdown(f"- [Coursera Course]({coursera})")
                    st.markdown(f"- [Udemy Course]({udemy})")

        # ── DEBUG ──
        with st.expander("DEBUG: Resume Chunks"):
            for i, ch in enumerate(chunks[:20]):
                st.text(f"{i}: {ch}")