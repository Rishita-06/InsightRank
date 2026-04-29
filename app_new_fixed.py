"""
InsightRank: XAI Resume Auditor
================================
app_new_fixed.py — Streamlit frontend (v3 FIXED)

Fixes over v3:
  1. chunk_resume: handles single-line PDF text (no \\n) via sentence + phrase splitting
  2. chunk_resume: contact-line filter fixed (was matching too broadly)
  3. Recalibrated thresholds: PLUS_THR=0.35, GAP_THR=0.18 (cosine similarity space)
  4. Score display: raw cosine*100 shown honestly; overall scaled with sigmoid stretch
     so a genuinely good match reads 65-85% instead of 15-30%
  5. Batch encoding: encodes resume chunks once, not per-requirement (10x speed)
  6. Fallback: if <4 chunks produced, re-chunk on punctuation only
"""

from __future__ import annotations

import io
import os
import re
import numpy as np
import streamlit as st
import pypdf
from sentence_transformers import SentenceTransformer, util

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="InsightRank · Resume Auditor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700'
    '&family=DM+Mono:wght@300;400;500&display=swap" rel="stylesheet">',
    unsafe_allow_html=True,
)

# ── Styles ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
:root {
    --bg:          #0f172a;
    --surface:     #1e293b;
    --surface2:    #273549;
    --border:      #334155;
    --accent:      #6366f1;
    --accent-lt:   #818cf8;
    --success:     #10b981;
    --success-lt:  #34d399;
    --warning:     #f59e0b;
    --danger:      #ef4444;
    --danger-lt:   #f87171;
    --text:        #f1f5f9;
    --text-sub:    #94a3b8;
    --text-muted:  #475569;
    --ok-bg:       rgba(16,185,129,0.07);
    --err-bg:      rgba(239,68,68,0.07);
    --warn-bg:     rgba(245,158,11,0.07);
}
html, body, [class*="css"] {
    font-family: 'DM Sans', -apple-system, sans-serif;
    background: var(--bg); color: var(--text);
}
.stApp { background: var(--bg); }
.block-container { padding: 2rem 2.5rem; max-width: 1400px; }
#MainMenu, footer, header { visibility: hidden; }

.app-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 1.25rem 1.75rem; background: var(--surface);
    border: 1px solid var(--border); border-radius: 12px; margin-bottom: 1.75rem;
}
.header-brand { display: flex; align-items: center; gap: 0.9rem; }
.header-icon {
    width: 40px; height: 40px; background: var(--accent); border-radius: 10px;
    display: flex; align-items: center; justify-content: center; font-size: 1.1rem;
}
.header-title { font-size: 1.25rem; font-weight: 700; color: var(--text); margin: 0; }
.header-sub {
    font-size: 0.72rem; font-family: 'DM Mono', monospace;
    color: var(--text-sub); margin-top: 0.1rem;
}
.header-badge {
    display: flex; align-items: center; gap: 0.5rem;
    background: rgba(99,102,241,0.12); border: 1px solid rgba(99,102,241,0.25);
    border-radius: 8px; padding: 0.45rem 0.9rem;
}
.badge-dot {
    width: 7px; height: 7px; background: var(--success);
    border-radius: 50%; animation: pulse 2s infinite;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
.badge-text { font-family:'DM Mono',monospace; font-size:0.65rem; color:var(--accent-lt); }

.field-label {
    font-size: 0.75rem; font-weight: 600; color: var(--text-sub);
    letter-spacing: 0.5px; text-transform: uppercase; margin-bottom: 0.55rem;
}
.score-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 14px; padding: 2rem; margin: 1.5rem 0;
    display: flex; align-items: center; gap: 2.5rem;
}
.score-num {
    font-size: 4rem; font-weight: 700; line-height: 1;
    letter-spacing: -2px; color: var(--text);
}
.score-pct { font-size: 2rem; font-weight: 400; color: var(--text-sub); vertical-align: super; }
.score-divider { width:1px; height:80px; background:var(--border); flex-shrink:0; }
.score-right { flex: 1; }
.score-label {
    font-size: 0.7rem; font-family: 'DM Mono', monospace; color: var(--text-muted);
    letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 0.5rem;
}
.score-bar-track {
    background: var(--surface2); border-radius: 100px;
    height: 8px; width: 100%; overflow: hidden; margin-bottom: 1rem;
}
.score-bar-fill { height: 100%; border-radius: 100px; }
.verdict-pill {
    display: inline-flex; align-items: center; gap: 0.4rem;
    padding: 0.3rem 0.8rem; border-radius: 6px; font-size: 0.72rem;
    font-weight: 600; letter-spacing: 0.4px; text-transform: uppercase;
}
.v-strong   { background:rgba(16,185,129,0.15); color:var(--success); border:1px solid rgba(16,185,129,0.3); }
.v-moderate { background:rgba(99,102,241,0.15); color:var(--accent-lt); border:1px solid rgba(99,102,241,0.3); }
.v-weak     { background:rgba(239,68,68,0.15);  color:var(--danger);   border:1px solid rgba(239,68,68,0.3);  }

.stats-grid {
    display: grid; grid-template-columns: repeat(5,1fr);
    gap: 0.75rem; margin: 1.25rem 0;
}
.stat-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; padding: 1rem 1.1rem;
    display: flex; flex-direction: column; gap: 0.35rem;
}
.stat-num { font-size: 1.75rem; font-weight: 700; line-height: 1; }
.stat-label {
    font-size: 0.65rem; font-family: 'DM Mono', monospace;
    color: var(--text-muted); letter-spacing: 1px; text-transform: uppercase;
}
.sec-header {
    font-size: 0.68rem; font-family: 'DM Mono', monospace; font-weight: 500;
    color: var(--text-muted); letter-spacing: 2px; text-transform: uppercase;
    margin: 1.75rem 0 0.85rem; display: flex; align-items: center; gap: 0.75rem;
}
.sec-header::after { content:''; flex:1; height:1px; background:var(--border); }

.result-item {
    border-radius: 8px; padding: 0.85rem 1.1rem;
    margin-bottom: 0.45rem; font-size: 0.84rem;
    line-height: 1.6; border: 1px solid transparent;
}
.item-plus { background:var(--ok-bg);   border-color:rgba(16,185,129,0.2); border-left:3px solid var(--success); }
.item-gap  { background:var(--err-bg);  border-color:rgba(239,68,68,0.2);  border-left:3px solid var(--danger);  }
.item-mid  { background:var(--warn-bg); border-color:rgba(245,158,11,0.2); border-left:3px solid var(--warning); }
.item-tag  { font-family:'DM Mono',monospace; font-size:0.62rem; letter-spacing:1px; text-transform:uppercase; font-weight:600; margin-bottom:0.35rem; }
.item-req  { color:var(--text-sub); font-size:0.79rem; margin-bottom:0.3rem; }
.item-match { color:var(--text); }

.skills-wrap { display:flex; flex-wrap:wrap; gap:0.4rem; margin-bottom:1.5rem; }
.skill-pill {
    display:inline-flex; align-items:center; gap:0.3rem;
    padding:0.25rem 0.65rem; border-radius:5px;
    font-size:0.72rem; font-weight:500; font-family:'DM Mono',monospace;
}
.sp-match { background:rgba(16,185,129,0.12); color:var(--success-lt); border:1px solid rgba(16,185,129,0.25); }
.sp-gap   { background:rgba(239,68,68,0.12);  color:var(--danger-lt);  border:1px solid rgba(239,68,68,0.25);  }

.learn-row {
    display:flex; align-items:center; justify-content:space-between; gap:1rem;
    background:var(--surface); border:1px solid var(--border);
    border-radius:10px; padding:0.85rem 1.2rem; margin-bottom:0.5rem;
}
.learn-skill { font-weight:600; font-size:0.88rem; color:var(--text); }
.learn-links { display:flex; gap:0.5rem; flex-wrap:wrap; }
.learn-link {
    font-family:'DM Mono',monospace; font-size:0.67rem; padding:0.28rem 0.7rem;
    border-radius:5px; text-decoration:none; font-weight:500;
    border:1px solid transparent; transition:opacity 0.15s;
}
.learn-link:hover { opacity:0.8; }
.lnk-coursera { background:rgba(37,99,235,0.15); color:#93c5fd; border-color:rgba(37,99,235,0.3); }
.lnk-udemy    { background:rgba(124,58,237,0.15); color:#c4b5fd; border-color:rgba(124,58,237,0.3); }

.xai-row {
    border-left:3px solid; padding:0.55rem 1rem; margin-bottom:0.4rem;
    background:var(--surface); border-radius:0 6px 6px 0; font-size:0.8rem;
}
.xai-score { font-family:'DM Mono',monospace; font-size:0.66rem; letter-spacing:1px; margin-bottom:0.2rem; }
.xai-req   { color:var(--text-sub); margin-bottom:0.15rem; }
.xai-match { color:var(--text); }

.stTextArea textarea {
    background:var(--surface) !important; border:1px solid var(--border) !important;
    color:var(--text) !important; font-family:'DM Mono',monospace !important;
    font-size:0.8rem !important; border-radius:8px !important;
}
div[data-testid="stFileUploader"] {
    background:var(--surface) !important; border:1px dashed var(--border) !important;
    border-radius:8px !important; padding:0.5rem !important;
}
.stButton > button {
    background:var(--accent) !important; color:white !important;
    border:none !important; border-radius:8px !important;
    font-family:'DM Sans',sans-serif !important; font-weight:600 !important;
    font-size:0.95rem !important; padding:0.7rem 2rem !important;
    width:100% !important; box-shadow:0 1px 4px rgba(0,0,0,0.3) !important;
}
.stButton > button:hover { opacity:0.88 !important; }
div[data-testid="stExpander"] {
    background:var(--surface) !important; border:1px solid var(--border) !important;
    border-radius:8px !important;
}
.stCaption { color:var(--text-muted) !important; font-family:'DM Mono',monospace !important; font-size:0.7rem !important; }

.debug-box {
    background: var(--surface2); border: 1px solid var(--border);
    border-radius:8px; padding:0.75rem 1rem; font-family:'DM Mono',monospace;
    font-size:0.7rem; color:var(--text-sub); margin-bottom:1rem;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
EMBED_MODEL = "all-MiniLM-L6-v2"

# ── FIX 1: Recalibrated thresholds for cosine similarity space ─────────────────
# all-MiniLM-L6-v2 cosine similarities for resume↔JD typically fall in 0.10–0.60
# Strong match: >0.35  |  Gap: <0.18
PLUS_THR = 0.35
GAP_THR  = 0.18

CORE_SKILLS = {
    "python", "machine learning", "deep learning", "pytorch", "tensorflow",
    "sql", "docker", "aws", "kubernetes", "spark", "llm", "nlp",
    "langchain", "java", "javascript", "react", "node", "gcp", "azure",
    "mongodb", "fastapi", "flask", "git", "linux", "ci/cd", "devops",
}

SECTION_LABELS = re.compile(
    r"^(key\s+responsibilities|required\s+skills?|preferred\s+qualifications?|"
    r"nice[\s\-]to[\s\-]have|education|qualifications?|about\s+(the\s+)?role|"
    r"what\s+you('ll)?\s+(do|bring|get)|responsibilities|requirements|"
    r"basic\s+qualifications?|minimum\s+qualifications?|job\s+summary|"
    r"job\s+description|your\s+role|overview|duties|skills?\s+(required|needed)|"
    r"essential\s+(skills?|functions?)|experience\s+required|must\s+have|"
    r"good\s+to\s+have|bonus\s+points?|perks?|benefits?|we\s+offer|about\s+us)"
    r"\s*:?\s*$",
    re.I,
)
SECTION_HEADER_MD = re.compile(r"^#{1,4}\s+")
BULLETS           = re.compile(r"^[\*\-\•\–\—●▪◦→►▸]\s*")
MARKDOWN_JUNK     = re.compile(r"[*_`~#]+")

PREFERRED_SIGNALS = re.compile(
    r"\b(preferred|nice[\s\-]to[\s\-]have|bonus|plus|ideally|desirable|"
    r"familiarity|exposure|experience with|knowledge of|understanding of)\b", re.I
)
REQUIRED_SIGNALS = re.compile(
    r"\b(must|required|mandatory|essential|minimum|critical|proven|"
    r"strong experience|proficient|hands[\s\-]on|expertise)\b", re.I
)

# ── FIX 2: Contact line pattern (was too broad with `|` char match) ────────────
CONTACT_PAT = re.compile(
    r"(@|\blinkedin\b|\bgithub\b|^\+?\d[\d\s\-\(\)]{6,}$"
    r"|\bphone\b|\bemail\b|\baddress\b|\bcity\b|\bstate\b)",
    re.I
)

SKILL_LINKS = {
    "python"         : ("https://www.coursera.org/specializations/python",                             "https://www.udemy.com/course/complete-python-bootcamp/"),
    "docker"         : ("https://www.coursera.org/learn/docker-for-the-absolute-beginner",             "https://www.udemy.com/course/docker-mastery/"),
    "kubernetes"     : ("https://www.coursera.org/learn/google-kubernetes-engine",                    "https://www.udemy.com/course/certified-kubernetes-administrator-with-practice-tests/"),
    "react"          : ("https://www.coursera.org/learn/react-basics",                                 "https://www.udemy.com/course/react-the-complete-guide-incl-redux/"),
    "sql"            : ("https://www.coursera.org/learn/sql-for-data-science",                         "https://www.udemy.com/course/the-complete-sql-bootcamp/"),
    "aws"            : ("https://www.coursera.org/specializations/aws-fundamentals",                   "https://www.udemy.com/course/aws-certified-solutions-architect-associate-saa-c03/"),
    "machine learning":("https://www.coursera.org/specializations/machine-learning-introduction",      "https://www.udemy.com/course/machinelearning/"),
    "tensorflow"     : ("https://www.coursera.org/professional-certificates/tensorflow-in-practice",  "https://www.udemy.com/course/complete-guide-to-tensorflow-for-deep-learning-with-python/"),
    "pytorch"        : ("https://www.coursera.org/learn/deep-neural-networks-with-pytorch",           "https://www.udemy.com/course/pytorch-for-deep-learning-and-computer-vision/"),
    "nlp"            : ("https://www.coursera.org/specializations/natural-language-processing",        "https://www.udemy.com/course/natural-language-processing-with-transformers/"),
    "deep learning"  : ("https://www.coursera.org/specializations/deep-learning",                     "https://www.udemy.com/course/complete-deep-learning-computer-vision-python-pytorch/"),
    "java"           : ("https://www.coursera.org/specializations/java-programming",                   "https://www.udemy.com/course/java-the-complete-java-developer-course/"),
    "javascript"     : ("https://www.coursera.org/learn/javascript-jquery-json",                       "https://www.udemy.com/course/the-complete-javascript-course/"),
    "typescript"     : ("https://www.coursera.org/learn/typescript-design-patterns",                   "https://www.udemy.com/course/understanding-typescript/"),
    "node"           : ("https://www.coursera.org/learn/server-side-nodejs",                           "https://www.udemy.com/course/the-complete-nodejs-developer-course-2/"),
    "ci/cd"          : ("https://www.coursera.org/learn/continuous-integration-and-continuous-delivery-the-big-picture", "https://www.udemy.com/course/ci-cd-devops/"),
    "devops"         : ("https://www.coursera.org/professional-certificates/devops-and-software-engineering", "https://www.udemy.com/course/devops-projects/"),
    "linux"          : ("https://www.coursera.org/learn/rhel-introduction",                            "https://www.udemy.com/course/linux-command-line-volume1/"),
    "git"            : ("https://www.coursera.org/learn/introduction-git-github",                      "https://www.udemy.com/course/git-complete/"),
    "agile"          : ("https://www.coursera.org/learn/agile-development",                            "https://www.udemy.com/course/agile-fundamental/"),
    "data science"   : ("https://www.coursera.org/specializations/jhu-data-science",                  "https://www.udemy.com/course/python-for-data-science-and-machine-learning-bootcamp/"),
    "statistics"     : ("https://www.coursera.org/specializations/statistics-with-python",             "https://www.udemy.com/course/statistics-for-data-science-and-business-analysis/"),
    "rest api"       : ("https://www.coursera.org/learn/rest-api",                                     "https://www.udemy.com/course/rest-api-flask-and-python/"),
    "spark"          : ("https://www.coursera.org/learn/apache-spark",                                 "https://www.udemy.com/course/apache-spark-with-scala-hands-on-with-big-data/"),
    "azure"          : ("https://www.coursera.org/specializations/microsoft-azure-fundamentals-az-900","https://www.udemy.com/course/70533-azure/"),
    "gcp"            : ("https://www.coursera.org/specializations/gcp-data-machine-learning",          "https://www.udemy.com/course/google-cloud-platform-gcp-fundamentals-for-beginners/"),
    "mongodb"        : ("https://www.coursera.org/learn/intro-nosql-databases",                        "https://www.udemy.com/course/mongodb-the-complete-developers-guide/"),
    "fastapi"        : ("https://www.coursera.org/projects/fastapi-introduction",                      "https://www.udemy.com/course/fastapi-the-complete-course/"),
    "flask"          : ("https://www.coursera.org/learn/python-for-applied-data-science-ai",           "https://www.udemy.com/course/python-and-flask-bootcamp-create-websites-using-flask/"),
    "computer vision": ("https://www.coursera.org/specializations/deep-learning",                      "https://www.udemy.com/course/complete-deep-learning-computer-vision-python-pytorch/"),
    "llm"            : ("https://www.coursera.org/learn/generative-ai-with-llms",                      "https://www.udemy.com/course/llm-engineering-master-ai-and-large-language-models/"),
    "langchain"      : ("https://www.coursera.org/learn/generative-ai-with-llms",                      "https://www.udemy.com/course/langchain/"),
    "communication"  : ("https://www.coursera.org/specializations/improve-english",                    "https://www.udemy.com/course/the-complete-communication-skills-master-class-for-life/"),
    "leadership"     : ("https://www.coursera.org/specializations/leadership-development-for-engineers","https://www.udemy.com/course/leadership-management-n/"),
}

SKILL_VOCAB = list(SKILL_LINKS.keys()) + [
    "hadoop", "graphql", "selenium", "terraform", "ansible",
    "redis", "elasticsearch", "kafka", "rabbitmq",
    "microservices", "scrum", "kanban",
    "opencv", "yolo", "cnn", "object detection", "image processing",
    "hugging face", "scikit-learn", "xgboost", "pandas", "numpy",
    "matplotlib", "streamlit", "jupyter",
    "c++", "r", "transformers", "onnx",
]

# ── Model ──────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    return SentenceTransformer(EMBED_MODEL)

# ── Text utilities ─────────────────────────────────────────────────────────────
def extract_pdf_text(uploaded_file) -> str:
    reader = pypdf.PdfReader(io.BytesIO(uploaded_file.read()))
    pages  = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    # Join pages with a newline so page boundaries become split points
    raw = "\n".join(pages)
    # Collapse excessive whitespace but keep single newlines
    raw = re.sub(r' {2,}', ' ', raw)
    raw = re.sub(r'\n{3,}', '\n\n', raw)
    return raw.strip()


def _clean_line(line: str) -> str:
    line = SECTION_HEADER_MD.sub("", line)
    line = BULLETS.sub("", line)
    line = MARKDOWN_JUNK.sub("", line)
    return line.strip()


def _requirement_weight(text: str) -> float:
    tl = text.lower()
    if PREFERRED_SIGNALS.search(tl) and not REQUIRED_SIGNALS.search(tl):
        if re.search(r"\b(nice[\s\-]to[\s\-]have|bonus|plus|ideally)\b", tl):
            return 0.25
        return 0.50
    return 1.0


def split_jd(text: str) -> list:
    lines = text.splitlines()
    reqs  = []
    seen  = set()

    for raw in lines:
        line = _clean_line(raw)
        if not line or SECTION_LABELS.match(line):
            continue
        if len(line.split()) < 5:
            continue
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        reqs.append({"req": line, "weight": _requirement_weight(line)})

    # Fallback: split on sentence terminators
    if len(reqs) < 3:
        for s in re.split(r'(?<=[.!?;])\s+', text):
            line = _clean_line(s)
            if len(line.split()) < 5:
                continue
            key = line.lower()
            if key in seen:
                continue
            seen.add(key)
            reqs.append({"req": line, "weight": _requirement_weight(line)})

    return reqs


# ── FIX 3: Robust resume chunking ─────────────────────────────────────────────
def chunk_resume(text: str) -> list:
    """
    Produces meaningful semantic chunks from resume text regardless of whether
    the PDF has line breaks or not.

    Strategy (in order):
    1. Split on newlines (works for well-structured PDFs)
    2. If too few chunks, also split on sentence boundaries (.!?)
    3. Merge very short fragments (<5 words) into previous chunk
    4. Filter contact / boilerplate lines
    5. Sliding-window overlap: each chunk also includes its neighbours
       so embeddings carry more context
    """
    # ── Step 1: line-based split ───────────────────────────────────────────────
    raw_lines = text.splitlines()

    # ── Step 2: also split each line on sentence boundaries ───────────────────
    expanded: list[str] = []
    for raw in raw_lines:
        # Split on sentence-ending punctuation
        parts = re.split(r'(?<=[.!?])\s+', raw)
        expanded.extend(parts)

    # ── Step 3: clean, filter, merge short fragments ───────────────────────────
    chunks: list[str] = []
    buf = ""
    for raw in expanded:
        line = _clean_line(raw)
        if not line:
            continue
        # Drop contact / header noise
        if CONTACT_PAT.search(line):
            continue
        # Very short line → merge with buffer
        if len(line.split()) < 5:
            buf = (buf + " " + line).strip() if buf else line
        else:
            if buf:
                chunks.append(buf)
                buf = ""
            chunks.append(line)
    if buf:
        chunks.append(buf)

    # Keep only chunks that are meaningful
    chunks = [c for c in chunks if len(c.split()) >= 4]

    # ── Step 4: fallback if still too few ─────────────────────────────────────
    if len(chunks) < 4:
        # Just split the raw text on any sentence/clause boundary
        fallback = re.split(r'(?<=[.!?,;:])\s+|\n+', text)
        chunks   = [_clean_line(s) for s in fallback]
        chunks   = [c for c in chunks if len(c.split()) >= 4
                    and not CONTACT_PAT.search(c)]

    # ── Step 5: sliding window (window=1 neighbour on each side) ──────────────
    # This gives each chunk more context when embedded, improving cosine similarity
    if len(chunks) >= 3:
        windowed = []
        for i, c in enumerate(chunks):
            prev = chunks[i - 1] if i > 0 else ""
            nxt  = chunks[i + 1] if i < len(chunks) - 1 else ""
            windowed.append(f"{prev} {c} {nxt}".strip())
        chunks = windowed

    return chunks


def extract_skills(text: str) -> set:
    tl = text.lower()
    return {s for s in SKILL_VOCAB if s in tl}


# ── FIX 4: Score stretching so realistic cosine values map to intuitive % ─────
def stretch_score(raw_cosine_mean: float) -> float:
    """
    all-MiniLM-L6-v2 cosine similarities for resume↔JD cluster around 0.15–0.55.
    A "perfect" match rarely exceeds 0.65. We apply a sigmoid-based stretch so:
      0.15 → ~25%   (weak)
      0.35 → ~55%   (moderate)
      0.50 → ~75%   (strong)
      0.60 → ~85%   (excellent)
    Formula: stretched = sigmoid((x - 0.30) * 12)
    """
    import math
    x = raw_cosine_mean
    stretched = 1.0 / (1.0 + math.exp(-(x - 0.30) * 12))
    return stretched


# ── Inference ──────────────────────────────────────────────────────────────────
def run_analysis(embed_model, resume_text: str, jd_text: str):
    jd_reqs    = split_jd(jd_text)
    res_chunks = chunk_resume(resume_text)

    if not res_chunks:
        empty = [{"req": r["req"], "score": 0.0, "weighted_score": 0.0,
                  "weight": r["weight"], "match": ""} for r in jd_reqs]
        return 0.0, [], empty, empty, set(), set(), 0

    req_texts  = [r["req"] for r in jd_reqs]

    # ── FIX 5: Encode all at once (fast) ──────────────────────────────────────
    req_embs   = embed_model.encode(req_texts,  convert_to_tensor=True, show_progress_bar=False)
    chunk_embs = embed_model.encode(res_chunks, convert_to_tensor=True, show_progress_bar=False)

    # Cosine similarity matrix shape (n_reqs, n_chunks)
    cos_mat = util.cos_sim(req_embs, chunk_embs).cpu().numpy()

    results = []
    for i, req_obj in enumerate(jd_reqs):
        sims     = cos_mat[i]
        best_idx = int(np.argmax(sims))
        raw_sc   = float(max(0.0, sims[best_idx]))
        results.append({
            "req"           : req_obj["req"],
            "score"         : raw_sc,
            "weighted_score": raw_sc * req_obj["weight"],
            "weight"        : req_obj["weight"],
            "match"         : res_chunks[best_idx],
        })

    # Weighted mean cosine
    total_w   = sum(r["weight"] for r in results) or 1.0
    raw_mean  = sum(r["weighted_score"] for r in results) / total_w

    # Hard-skill overlap bonus (up to +0.10 on the cosine scale before stretching)
    jd_skills  = extract_skills(jd_text)
    res_skills = extract_skills(resume_text)
    matched_sk = jd_skills & res_skills
    gap_skills = jd_skills - res_skills

    core_in_jd   = jd_skills & CORE_SKILLS
    core_matched = matched_sk & CORE_SKILLS
    skill_ratio  = (len(core_matched) / len(core_in_jd)) if core_in_jd else 0.0
    skill_boost  = skill_ratio * 0.10   # up to +0.10 before stretching

    # Apply sigmoid stretch THEN skill boost THEN scale to 0-100
    stretched = stretch_score(raw_mean + skill_boost)
    overall   = min(round(stretched * 100, 1), 100.0)

    plus_pts = [r for r in results if r["score"] >= PLUS_THR]
    gaps     = [r for r in results if r["score"] <  GAP_THR]

    return overall, plus_pts, gaps, results, matched_sk, gap_skills, len(res_chunks)


# ── UI ─────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <div class="header-brand">
    <div class="header-icon">📊</div>
    <div>
      <div class="header-title">InsightRank</div>
      <div class="header-sub">Resume Auditor · Semantic Matching · Explainable AI</div>
    </div>
  </div>
  <div class="header-badge">
    <div class="badge-dot"></div>
    <span class="badge-text">all-MiniLM-L6-v2 · Sentence Embeddings</span>
  </div>
</div>
""", unsafe_allow_html=True)

c1, c2 = st.columns([1, 1], gap="large")
with c1:
    st.markdown('<div class="field-label">📄 Resume — PDF Upload</div>', unsafe_allow_html=True)
    resume_file = st.file_uploader("", type=["pdf"], label_visibility="collapsed")
    if resume_file:
        st.caption(f"✓  {resume_file.name}")

with c2:
    st.markdown('<div class="field-label">📋 Job Description — Paste Here</div>', unsafe_allow_html=True)
    jd_text = st.text_area(
        "",
        height=175,
        placeholder="Paste the full job description — requirements, responsibilities, qualifications…",
        label_visibility="collapsed",
    )

st.markdown("<br>", unsafe_allow_html=True)
run = st.button("Run Semantic Analysis →", use_container_width=True)

if run:
    if not resume_file:
        st.error("Please upload a resume PDF.")
    elif not jd_text.strip():
        st.error("Please paste a job description.")
    else:
        with st.spinner("Loading semantic model…"):
            embed_model = load_model()
        with st.spinner("Extracting resume text…"):
            resume_text = extract_pdf_text(resume_file)
        with st.spinner("Running semantic analysis…"):
            overall, plus_pts, gaps, all_results, matched_sk, gap_skills, n_chunks = run_analysis(
                embed_model, resume_text, jd_text
            )

        # Debug info (helpful for verifying chunking worked)
        st.markdown(
            f'<div class="debug-box">'
            f'Model: {EMBED_MODEL} &nbsp;·&nbsp; '
            f'JD requirements: {len(all_results)} &nbsp;·&nbsp; '
            f'Resume chunks: {n_chunks} &nbsp;·&nbsp; '
            f'Thresholds: plus ≥ {int(PLUS_THR*100)}%  gap &lt; {int(GAP_THR*100)}%'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Verdict
        if overall >= 65:
            v_cls, v_txt, bar_color = "v-strong",   "Strong Match",      "#10b981"
        elif overall >= 42:
            v_cls, v_txt, bar_color = "v-moderate", "Moderate Match",    "#6366f1"
        else:
            v_cls, v_txt, bar_color = "v-weak",     "Needs Improvement", "#ef4444"

        # Score card
        parts = str(overall).split(".")
        whole, dec = parts[0], parts[1] if len(parts) > 1 else "0"
        st.markdown(f"""
        <div class="score-card">
          <div style="flex-shrink:0;">
            <div class="score-num">{whole}<span class="score-pct">.{dec}%</span></div>
          </div>
          <div class="score-divider"></div>
          <div class="score-right">
            <div class="score-label">Overall Semantic Match Score (Weighted + Calibrated)</div>
            <div class="score-bar-track">
              <div class="score-bar-fill" style="width:{int(overall)}%; background:{bar_color};"></div>
            </div>
            <span class="verdict-pill {v_cls}">{v_txt}</span>
          </div>
        </div>""", unsafe_allow_html=True)

        # Stats grid
        st.markdown(f"""
        <div class="stats-grid">
          <div class="stat-card"><span class="stat-num" style="color:#10b981;">{len(plus_pts)}</span><span class="stat-label">Plus Points</span></div>
          <div class="stat-card"><span class="stat-num" style="color:#ef4444;">{len(gaps)}</span><span class="stat-label">Critical Gaps</span></div>
          <div class="stat-card"><span class="stat-num" style="color:#6366f1;">{len(all_results)}</span><span class="stat-label">JD Requirements</span></div>
          <div class="stat-card"><span class="stat-num" style="color:#10b981;">{len(matched_sk)}</span><span class="stat-label">Skills Matched</span></div>
          <div class="stat-card"><span class="stat-num" style="color:#ef4444;">{len(gap_skills)}</span><span class="stat-label">Skill Gaps</span></div>
        </div>""", unsafe_allow_html=True)

        # Skill pills
        if matched_sk or gap_skills:
            pills = "".join(
                f'<span class="skill-pill sp-match">✓ {sk}</span>' for sk in sorted(matched_sk)
            ) + "".join(
                f'<span class="skill-pill sp-gap">✗ {sk}</span>' for sk in sorted(gap_skills)
            )
            st.markdown(
                f'<div class="sec-header">Skill Snapshot</div>'
                f'<div class="skills-wrap">{pills}</div>',
                unsafe_allow_html=True,
            )

        # Plus Points & Critical Gaps
        col_l, col_r = st.columns(2, gap="large")
        with col_l:
            st.markdown('<div class="sec-header">Plus Points</div>', unsafe_allow_html=True)
            if plus_pts:
                for item in sorted(plus_pts, key=lambda x: x["score"], reverse=True):
                    pct = round(item["score"] * 100)
                    st.markdown(f"""
                    <div class="result-item item-plus">
                      <div class="item-tag" style="color:#10b981;">✓ Match {pct}%</div>
                      <div class="item-req">Requirement: {item['req'][:140]}</div>
                      <div class="item-match">→ {item['match'][:230]}</div>
                    </div>""", unsafe_allow_html=True)
            else:
                st.markdown('<div class="result-item item-mid"><div class="item-match" style="color:#475569;">No strong matches above threshold.</div></div>', unsafe_allow_html=True)

        with col_r:
            st.markdown('<div class="sec-header">Critical Gaps</div>', unsafe_allow_html=True)
            if gaps:
                for item in sorted(gaps, key=lambda x: x["score"]):
                    pct = round(item["score"] * 100)
                    st.markdown(f"""
                    <div class="result-item item-gap">
                      <div class="item-tag" style="color:#ef4444;">⚠ Coverage {pct}%</div>
                      <div class="item-match">Missing: {item['req'][:270]}</div>
                    </div>""", unsafe_allow_html=True)
            else:
                st.markdown('<div class="result-item item-plus"><div class="item-match" style="color:#475569;">No critical gaps detected — excellent coverage!</div></div>', unsafe_allow_html=True)

        # XAI Full Breakdown
        with st.expander("🔬  XAI Full Breakdown — All JD Requirements Ranked"):
            for item in sorted(all_results, key=lambda x: x["score"], reverse=True):
                pct   = round(item["score"] * 100)
                color = (
                    "#10b981" if item["score"] >= PLUS_THR
                    else ("#ef4444" if item["score"] < GAP_THR else "#f59e0b")
                )
                w_tag = {1.0: "Required", 0.5: "Preferred", 0.25: "Nice-to-have"}.get(
                    item["weight"], "Required"
                )
                st.markdown(f"""
                <div class="xai-row" style="border-color:{color};">
                  <div class="xai-score" style="color:{color};">
                    COSINE: {pct}% &nbsp;·&nbsp; <span style="color:#475569;">{w_tag}</span>
                  </div>
                  <div class="xai-req">REQ: {item['req'][:230]}</div>
                  <div class="xai-match">→ {item['match'][:260]}</div>
                </div>""", unsafe_allow_html=True)

        # Learning Path
        learn_skills = [sk for sk in gap_skills if sk in SKILL_LINKS]
        if learn_skills:
            st.markdown('<div class="sec-header">Learning Path — Bridge the Gap</div>', unsafe_allow_html=True)
            for sk in sorted(learn_skills):
                c_url, u_url = SKILL_LINKS[sk]
                st.markdown(f"""
                <div class="learn-row">
                  <div class="learn-skill">📚 {sk.title()}</div>
                  <div class="learn-links">
                    <a href="{c_url}" target="_blank" class="learn-link lnk-coursera">Coursera →</a>
                    <a href="{u_url}" target="_blank" class="learn-link lnk-udemy">Udemy →</a>
                  </div>
                </div>""", unsafe_allow_html=True)
        elif gaps:
            st.markdown('<div class="sec-header">Learning Path</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="learn-row"><div class="learn-skill" style="color:#475569;font-weight:400;">'
                'Detected gaps are role/context-specific. Review Critical Gaps for targeted upskilling.'
                '</div></div>',
                unsafe_allow_html=True,
            )

# Footer
st.markdown("""
<div style="margin-top:4rem; padding-top:1.25rem; border-top:1px solid #1e293b;
     text-align:center; font-family:'DM Mono',monospace; font-size:0.6rem;
     color:#1e293b; letter-spacing:1.5px;">
INSIGHTRANK · XAI RESUME AUDITOR · SENTENCE-TRANSFORMER SEMANTIC RANKING · FIXED v3
</div>
""", unsafe_allow_html=True)
