# RAG Desk (Local) — Bilingual RAG Desktop Web App

A privacy-first, local Retrieval-Augmented Generation (RAG) demo built with **Streamlit + ChromaDB + SentenceTransformers**.  
Users can drop documents into `/docs`, rebuild the index in the browser, and ask questions with **citations**.  
Supports **English & 中文** UI/answers.

## ✨ Features
- Local-first RAG (no cloud required)
- Drag-and-drop docs into `/docs` (txt/pdf/docx supported in code)
- One-click **Build / Rebuild Index**
- Bilingual mode: **English / 中文**
- Citation-style references (file/page/chunk)
- Windows-friendly launcher (`Start_RAGDesk.bat`)

## 🧠 How it works
1. Load documents from `/docs`
2. Chunk text and create embeddings (SentenceTransformers)
3. Store vectors in ChromaDB (local folder `chroma_db/`)
4. Retrieve top-k relevant chunks
5. Generate a concise answer + show citations

## 🖥️ Demo
English:
![English Demo]

中文:
![Chinese Demo]

(Optional) GIF:
![Demo GIF]

## 🚀 Quick Start (Windows)
### Option A: Double-click
1. Install Anaconda / Miniconda
2. Double-click `Start_RAGDesk.bat`
3. Open: http://localhost:8501
4. Put docs into `docs/` → click **Build / Rebuild Index** → ask questions

### Option B: Command line
```bash
conda activate ragdesk
cd %USERPROFILE%\Desktop\ragdesk-app
pip install -r requirements.txt
python -m streamlit run app.py
