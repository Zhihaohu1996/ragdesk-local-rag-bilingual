import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple

import streamlit as st

import re

def _has_cjk(text: str) -> bool:
    return bool(re.search(r'[\u4e00-\u9fff]', text or ''))

def _has_latin(text: str) -> bool:
    return bool(re.search(r'[A-Za-z]', text or ''))

from pypdf import PdfReader
from docx import Document

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

_TRANSLATION_OK = True
try:
    from transformers import pipeline
except Exception:
    _TRANSLATION_OK = False

APP_TITLE = "RAG Desk (Local)"

# Use the folder where this app.py lives as a stable base.
# This avoids issues when users launch Streamlit from a different working directory.
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DOCS_DIR = "docs"  # user-facing default
CHROMA_DIR = str(BASE_DIR / "chroma_db")
COLLECTION_NAME = "ragdesk"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

CHUNK_SIZE = 700
CHUNK_OVERLAP = 120


def resolve_user_path(path_str: str) -> Path:
    """Resolve a user-provided path.

    - If the user types a relative path (e.g. "docs"), resolve it relative to app.py.
    - If the user types an absolute path, use it as-is.
    """
    p = Path(path_str.strip().strip('"').strip("'"))
    if not p.is_absolute():
        p = (BASE_DIR / p)
    return p


def list_docs_status(docs_dir: Path) -> List[Dict[str, Any]]:
    """Return file list for the UI with last modified time."""
    rows: List[Dict[str, Any]] = []
    if not docs_dir.exists() or not docs_dir.is_dir():
        return rows
    for p in sorted(docs_dir.glob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".txt", ".pdf", ".docx"}:
            continue
        st_mtime = p.stat().st_mtime
        rows.append(
            {
                "file": p.name,
                "type": p.suffix.lower().lstrip("."),
                "size_kb": round(p.stat().st_size / 1024, 1),
                "modified": datetime.fromtimestamp(st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                "_mtime": st_mtime,
            }
        )
    return rows

def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def read_pdf(path: Path) -> List[Dict[str, Any]]:
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append({"text": text, "page": i})
    return pages

def read_docx(path: Path) -> List[Dict[str, Any]]:
    doc = Document(str(path))
    text = "\n".join([p.text for p in doc.paragraphs if p.text is not None])
    return [{"text": text, "page": 0}]

def iter_docs(docs_dir: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for p in sorted(docs_dir.glob("*")):
        if not p.is_file():
            continue
        suffix = p.suffix.lower()
        if suffix == ".txt":
            items.append({"filename": p.name, "filetype": "text", "pages": [{"text": read_txt(p), "page": 0}]})
        elif suffix == ".pdf":
            items.append({"filename": p.name, "filetype": "pdf", "pages": read_pdf(p)})
        elif suffix == ".docx":
            items.append({"filename": p.name, "filetype": "docx", "pages": read_docx(p)})
    return items

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    text = (text or "").replace("\r\n", "\n").strip()
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks

@st.cache_resource(show_spinner=False)
def get_chroma_client():
    os.makedirs(CHROMA_DIR, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_DIR)


@st.cache_resource(show_spinner=False)
def get_embedding_fn():
    # Loading the embedding model can take a bit of time; cache it.
    return SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)


def get_collection():
    # Don't cache the collection object. If we rebuild (delete/create) the collection,
    # a cached handle can become stale and show a wrong count (often "0").
    client = get_chroma_client()
    emb_fn = get_embedding_fn()
    return client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=emb_fn)

def rebuild_index(docs_dir: Path) -> Tuple[int, int]:
    client = get_chroma_client()
    emb_fn = get_embedding_fn()

    # Clean rebuild: delete the whole collection (if it exists), then recreate.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    col = client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=emb_fn)

    docs = iter_docs(docs_dir)
    ids, documents, metadatas = [], [], []
    for d in docs:
        for page in d["pages"]:
            for ch_i, ch in enumerate(chunk_text(page["text"], CHUNK_SIZE, CHUNK_OVERLAP)):
                ids.append(f"{d['filename']}|p{page['page']}|c{ch_i}")
                documents.append(ch)
                metadatas.append({
                    "filename": d["filename"],
                    "filetype": d["filetype"],
                    "page": int(page["page"]),
                    "chunk_index": int(ch_i),
                })

    if ids:
        col.add(ids=ids, documents=documents, metadatas=metadatas)
    return len(ids), len(docs)

def retrieve(question: str, top_k: int = 5):
    col = get_collection()
    res = col.query(query_texts=[question], n_results=top_k)
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    out = []
    for d, m in zip(docs, metas):
        if d and m:
            out.append((d, m))
    return out

@st.cache_resource
def get_translators():
    if not _TRANSLATION_OK:
        return None, None
    try:
        zh2en = pipeline("translation", model="Helsinki-NLP/opus-mt-zh-en")
        en2zh = pipeline("translation", model="Helsinki-NLP/opus-mt-en-zh")
        return zh2en, en2zh
    except Exception:
        return None, None

def translate_text(text: str, target_lang: str) -> str:
    """Translate only when it likely helps.

    - target=en: translate *Chinese-only* snippets to English.
    - target=zh: translate *English-only* snippets to Chinese.

    Mixed-language strings (both CJK + Latin) are left untouched to avoid garbling.
    """
    zh2en, en2zh = get_translators()
    text = (text or "").strip()
    if not text:
        return text

    has_cjk = _has_cjk(text)
    has_latin = _has_latin(text)

    if target_lang == "en":
        # Already English-ish, or mixed: don't translate.
        if not has_cjk or has_latin:
            return text
        if zh2en is None:
            return text
        try:
            return zh2en(text, max_length=512)[0]["translation_text"]
        except Exception:
            return text

    if target_lang == "zh":
        # Already Chinese-ish, or mixed: don't translate.
        if not has_latin or has_cjk:
            return text
        if en2zh is None:
            return text
        try:
            return en2zh(text, max_length=512)[0]["translation_text"]
        except Exception:
            return text

    return text

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("Local RAG demo (ChromaDB + SentenceTransformers). Put .txt/.pdf/.docx into docs/ and build the index.")

with st.sidebar:
    st.header("Settings")
    lang = st.selectbox("Language / 语言", ["English", "中文"], index=0)
    LANG = "en" if lang == "English" else "zh"

    docs_dir_str = st.text_input("Docs folder", value=DEFAULT_DOCS_DIR)
    docs_dir = resolve_user_path(docs_dir_str)
    st.caption(f"Resolved docs path: `{docs_dir}`")

    # Show what the app can actually see inside docs/
    def _docs_rows(folder: Path) -> List[Dict[str, Any]]:
        if not folder.exists() or not folder.is_dir():
            return []
        rows: List[Dict[str, Any]] = []
        for p in sorted(folder.iterdir()):
            if not p.is_file():
                continue
            suffix = p.suffix.lower()
            if suffix not in {".txt", ".pdf", ".docx"}:
                continue
            try:
                mtime = datetime.fromtimestamp(p.stat().st_mtime)
                mtime_str = mtime.strftime("%Y-%m-%d %H:%M:%S")
                size_kb = round(p.stat().st_size / 1024, 1)
            except Exception:
                mtime_str = "?"
                size_kb = "?"
            rows.append({
                "file": p.name,
                "type": suffix.lstrip("."),
                "modified": mtime_str,
                "size_kb": size_kb,
            })
        return rows

    rows = _docs_rows(docs_dir)
    if rows:
        newest = max(r["modified"] for r in rows if r.get("modified") not in {None, "?"})
        st.write(f"Detected docs: **{len(rows)}**  |  Latest update: **{newest}**")
        st.dataframe(rows, hide_index=True, use_container_width=True)
    else:
        st.info("No .txt/.pdf/.docx detected in docs folder.")

    top_k = st.slider("Top-K references", 1, 10, 5, 1)

    st.divider()
    st.subheader("Index")
    col = get_collection()
    try:
        count = col.count()
    except Exception:
        count = 0
    st.write(f"Indexed chunks: **{count}**")
    st.caption(f"ChromaDB: `{Path(CHROMA_DIR)}`  |  Collection: `{COLLECTION_NAME}`")

    if "last_indexed_at" not in st.session_state:
        st.session_state["last_indexed_at"] = None
    if st.session_state["last_indexed_at"]:
        st.write(f"Last indexed: **{st.session_state['last_indexed_at']}**")

    if st.button("Build / Rebuild Index", type="primary"):
        if not docs_dir.exists():
            st.error(f"Folder not found: {docs_dir.resolve()}")
        else:
            with st.spinner("Indexing documents..."):
                added, files = rebuild_index(docs_dir)
            st.session_state["last_indexed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.success(f"Done. Files: {files}, chunks: {added}")
            st.rerun()

    st.divider()
    st.subheader("Offline translation (optional)")
    if _TRANSLATION_OK:
        zh2en, en2zh = get_translators()
        if (zh2en is None) or (en2zh is None):
            st.warning("Translator not ready. Run: pip install -U transformers torch sentencepiece")
        else:
            st.success("Translator ready.")
    else:
        st.info("Transformers not installed. Translation disabled.")

question = st.text_input("Ask a question / 提问", value="What is the return policy?")

if st.button("Search", type="secondary"):
    col = get_collection()
    try:
        count = col.count()
    except Exception:
        count = 0
    if count == 0:
        st.error("Index is empty. Click **Build / Rebuild Index** first.")
    else:
        with st.spinner("Retrieving..."):
            pairs = retrieve(question, top_k=top_k)

        if not pairs:
            st.warning("No results.")
        else:
            st.subheader("Answer" if LANG == "en" else "回答")
            bullets = []
            for i, (doc, meta) in enumerate(pairs, start=1):
                snippet = (doc or "").replace("\n", " ").strip()
                if len(snippet) > 260:
                    snippet = snippet[:260] + "…"
                if LANG == "en":
                    snippet = translate_text(snippet, "en")
                else:
                    snippet = translate_text(snippet, "zh")
                bullets.append((i, snippet, meta))

            for i, snippet, meta in bullets:
                st.markdown(f"- {snippet}  **[{i}]**")

            st.subheader("Sources" if LANG == "en" else "引用来源")
            for i, _, meta in bullets:
                st.markdown(
                    f"**[{i}]** {meta.get('filename')} | type={meta.get('filetype')} | page={meta.get('page')} | chunk={meta.get('chunk_index')}"
                )

st.divider()
st.caption("Tip: English questions can retrieve Chinese docs (multilingual embeddings).")
