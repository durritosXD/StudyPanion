# app.py
import os
from dotenv import load_dotenv
import streamlit as st
import google.generativeai as genai

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

st.set_page_config(page_title="Legal Contract Assistant", page_icon="⚖️", layout="wide")
st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #0b1220 0%, #101827 40%, #0f172a 100%); color: #e5e7eb; }
.block-container{ padding-top: 1.2rem; padding-bottom: 2rem;}
section[data-testid="stSidebar"] { background:#0f172a; border-right:1px solid #1f2937; }
.sidebar-card{ background:#111827; border:1px solid #1f2937; border-radius:14px; padding:16px; box-shadow:0 8px 18px rgba(0,0,0,.35);}
.hero{ padding:16px 24px; border-radius:18px; background:
  radial-gradient(1200px 300px at 10% -20%, rgba(34,211,238,.20), transparent 60%),
  radial-gradient(900px 260px at 120% -10%, rgba(99,102,241,.20), transparent 60%), #0b1220;
  border:1px solid #1f2937; box-shadow:0 12px 26px rgba(0,0,0,.35);}
.hero h1{ margin:0 0 6px 0; background:linear-gradient(90deg,#22d3ee,#8b5cf6);
  -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent; font-weight:900;}
.file-pill{ display:inline-block; background:#0b1220; border:1px solid #1f2937; padding:6px 10px; border-radius:999px; margin:4px 6px 0 0;}
.stTextInput > div > div > input{ background:#0b1220; border:1.5px solid #334155; color:#e5e7eb; border-radius:12px; padding:12px 14px;}
button[kind="primary"]{ background:linear-gradient(90deg,#22d3ee,#6366f1); color:white; border-radius:12px; font-weight:700; border:none;}
button[kind="primary"]:hover{ filter:brightness(1.05); box-shadow:0 6px 18px rgba(99,102,241,.45);}
.card{ background:#0b1220; border:1px solid #1f2937; border-radius:16px; padding:14px 16px; box-shadow:0 10px 22px rgba(0,0,0,.35); }
.chat-wrap{ padding:6px 4px;}
.user-bubble,.bot-bubble{ max-width:80%; padding:12px 16px; margin:6px 0; border-radius:16px; line-height:1.45;
  word-wrap:break-word; white-space:pre-wrap; border:1px solid #1f2937;}
.user-bubble{ margin-left:auto; background:#1d4ed8; color:#e5e7eb;}
.bot-bubble{ margin-right:auto; background:#111827; color:#e5e7eb;}
.citation{ color:#93c5fd; font-size:.9rem; }
.small-dim{ color:#94a3b8; font-size:.9rem;}
.footer{ color:#94a3b8; font-size:.85rem; text-align:center; margin-top:18px; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# ENV + GOOGLE CONFIG
# -----------------------------------------------------------------------------
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")
genai.configure(api_key=api_key)

# -----------------------------------------------------------------------------
# LEGAL RAG HELPERS
# -----------------------------------------------------------------------------
SPLITTER = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=250)

def extract_chunks_with_metadata(uploaded_files):
   
    texts, metadatas = [], []
    per_page_cache = []  # For risk scan (file, page, text)

    for file in uploaded_files:
        fname = getattr(file, "name", "uploaded.pdf")
        try:
            reader = PdfReader(file)
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    per_page_cache.append({"source": fname, "page": i+1, "text": page_text})
                    chunks = SPLITTER.split_text(page_text)
                    for ch in chunks:
                        texts.append(ch)
                        metadatas.append({"source": fname, "page": i+1})
        except Exception as e:
            st.error(f"Failed to read {fname}: {e}")

    return texts, metadatas, per_page_cache

def build_or_load_index(texts, metadatas):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vs = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)
    vs.save_local("faiss_index")
    return True

def legal_prompt():
    return PromptTemplate(
        template=(
            "You are a precise legal analyst. Use ONLY the provided context to answer.\n"
            "If the answer isn't present, say: 'Answer is not available in the context.'\n"
            "Write clearly and concisely for a non-lawyer reader.\n\n"
            "Context:\n{context}\n\n"
            "Question:\n{question}\n\n"
            "Answer (be specific, include clause names if present):"
        ),
        input_variables=["context", "question"],
    )

def build_chain(model_name: str):
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.2)
    return load_qa_chain(llm, chain_type="stuff", prompt=legal_prompt())

def query_with_citations(question: str, model_name: str):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vs = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = vs.similarity_search(question, k=4)  # get top 4 chunks
    chain = build_chain(model_name)
    out = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
    answer = out.get("output_text", "")

    # collect citations (unique by (source, page))
    cites = []
    seen = set()
    for d in docs:
        src = d.metadata.get("source", "document")
        page = d.metadata.get("page", "?")
        key = (src, page)
        if key not in seen:
            seen.add(key)
            cites.append(f"{src} p.{page}")
    return answer, cites

# Simple keyword-based risk scanner (lightweight)
RISK_PATTERNS = {
    "Indemnity": ["indemnify", "indemnification", "hold harmless"],
    "Limitation of Liability": ["limitation of liability", "liability cap", "consequential damages"],
    "Auto-Renewal": ["auto-renew", "automatic renewal", "renews automatically"],
    "Termination": ["terminate for convenience", "termination for cause", "early termination fee"],
    "Exclusivity": ["exclusive", "exclusivity", "non-compete"],
    "Penalties/Liquidated Damages": ["liquidated damages", "penalty", "late fee"],
    "Confidentiality": ["confidential", "non-disclosure", "NDA", "confidential information"],
    "Governing Law/Dispute": ["governing law", "jurisdiction", "venue", "arbitration"],
    "Assignment": ["assignment", "may not assign", "transfer of rights"],
    "IP Ownership": ["intellectual property", "IP ownership", "work product", "inventions"]
}

def detect_risks(per_page_cache):
    found = []  # list of (label, file, page, short_snippet)
    for page in per_page_cache:
        lower = page["text"].lower()
        for label, keywords in RISK_PATTERNS.items():
            if any(k in lower for k in keywords):
                snippet = page["text"][:220].replace("\n", " ")
                found.append((label, page["source"], page["page"], snippet + ("..." if len(page["text"])>220 else "")))
    return found

# -----------------------------------------------------------------------------
# SESSION STATE
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "indexed" not in st.session_state:
    st.session_state.indexed = False
if "uploaded_names" not in st.session_state:
    st.session_state.uploaded_names = []
if "per_page_cache" not in st.session_state:
    st.session_state.per_page_cache = []

# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("<div class='sidebar-card'>", unsafe_allow_html=True)
    st.title("📂 Legal Docs")
    st.caption("Upload contracts, NDAs, MSAs, policies… then process & ask questions.")

    model_choice = st.selectbox(
        "LLM Model",
        options=["gemini-1.5-flash", "gemini-1.5-pro"],
        index=0,
        help="Flash is faster; Pro is stronger at reasoning."
    )

    files = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)
    if st.button("🚀 Submit & Process", use_container_width=True):
        if not files:
            st.warning("Please upload at least one PDF.")
        else:
            with st.spinner("Extracting, chunking, embedding, and indexing…"):
                texts, metas, cache = extract_chunks_with_metadata(files)
                build_or_load_index(texts, metas)
                st.session_state.indexed = True
                st.session_state.per_page_cache = cache
                st.session_state.uploaded_names = [getattr(f, "name", "uploaded.pdf") for f in files]
            st.success("Indexed! Ask legal questions on the right.")

    if st.button("🧹 Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.success("Chat cleared.")
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
st.markdown("""
<div class='hero'>
  <h1>⚖️ Legal Contract Assistant</h1>
  <p>Ask precise questions about your contracts. Answers cite files & pages. (This tool is not legal advice.)</p>
</div>
""", unsafe_allow_html=True)

# Show which files are indexed
if st.session_state.uploaded_names:
    st.markdown("#### 📄 Indexed Files")
    st.markdown("".join([f"<span class='file-pill'>{n}</span>" for n in st.session_state.uploaded_names]), unsafe_allow_html=True)

# Quick legal actions
st.markdown("### ⚡ Quick Legal Questions")
qcols = st.columns(5)
quick_map = {
    "Termination clause": "Locate and explain the termination clause.",
    "Payment terms": "Summarize the payment terms, amounts, due dates, and late fees.",
    "Governing law": "What is the governing law and dispute resolution mechanism?",
    "IP ownership": "Who owns intellectual property/work product created under this agreement?",
    "Confidentiality": "Summarize the confidentiality obligations and exceptions."
}
clicked = None
for i, (label, prompt) in enumerate(quick_map.items()):
    if qcols[i % 5].button(label):
        clicked = prompt
if clicked:
    st.session_state.messages.append({"role": "user", "content": clicked})
    if not st.session_state.indexed:
        st.warning("Upload PDFs and click Submit & Process first.")
    else:
        with st.spinner("Thinking…"):
            ans, cites = query_with_citations(clicked, model_choice)
        st.session_state.messages.append({"role": "assistant", "content": ans, "citations": cites})

# Free-form question
st.markdown("### ❓ Ask a Contract Question")
user_q = st.text_input("e.g., Are there penalties for early termination? Provide details.")
if user_q:
    st.session_state.messages.append({"role": "user", "content": user_q})
    if not st.session_state.indexed:
        st.warning("Upload PDFs and click Submit & Process first.")
    else:
        with st.spinner("Thinking…"):
            ans, cites = query_with_citations(user_q, model_choice)
        st.session_state.messages.append({"role": "assistant", "content": ans, "citations": cites})

# One-click summary
st.markdown("### 🧭 Key Clauses Summary")
if st.button("Generate a summary of key clauses"):
    if not st.session_state.indexed:
        st.warning("Upload PDFs and click Submit & Process first.")
    else:
        prompt = ("Provide a concise bullet-point summary of the key clauses "
                  "(parties, scope, term, termination, payment, confidentiality, IP, liability, indemnity, governing law, assignment, renewal). "
                  "Be specific and actionable.")
        st.session_state.messages.append({"role": "user", "content": "Summarize key clauses."})
        with st.spinner("Summarizing…"):
            ans, cites = query_with_citations(prompt, model_choice)
        st.session_state.messages.append({"role": "assistant", "content": ans, "citations": cites})

# Risk flags panel
st.markdown("### 🚩 Potential Risk Flags (keyword scan)")
if st.session_state.per_page_cache:
    findings = detect_risks(st.session_state.per_page_cache)
    if not findings:
        st.info("No common red-flag keywords detected. (This is a simple keyword scan; not legal advice.)")
    else:
        for label, src, page, snippet in findings[:30]:
            st.markdown(f"**{label}** — *{src} p.{page}*  \n<span class='small-dim'>{snippet}</span>", unsafe_allow_html=True)

# Conversation
st.markdown("### 💬 Conversation")
st.markdown("<div class='card chat-wrap'>", unsafe_allow_html=True)
if not st.session_state.messages:
    st.markdown("<div class='bot-bubble'>Hi! Upload contract PDFs, click <b>Submit & Process</b>, then use quick questions or ask anything.</div>", unsafe_allow_html=True)
else:
    for msg in st.session_state.messages:
        cls = "user-bubble" if msg["role"] == "user" else "bot-bubble"
        safe = msg["content"].replace("<", "&lt;").replace(">", "&gt;")
        st.markdown(f"<div class='{cls}'>{safe}</div>", unsafe_allow_html=True)
        if msg.get("citations"):
            st.markdown(f"<div class='citation'>Sources: {', '.join(msg['citations'])}</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='footer'>This tool is for information only and is not legal advice.</div>", unsafe_allow_html=True)
