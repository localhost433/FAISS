# Vibe Search Engine (FAISS + FastAPI)

A local semantic search engine for places, with:
- Sentence-Transformers for embeddings
- FAISS for fast similarity search
- FastAPI for backend
- JS + HTML/CSS for frontend

---

## Structure
The project should look like this:
- **backend/**
  - **app/**
    - `__init__.py`
    - `main.py` — App setup
    - **core/** — Logic helper functions
      - `config.py` — Paths/settings
      - `embeddings.py` — Load model, embeddings
      - `faiss_io.py` — FAISS indices
      - `text_utils.py` — Preprocess
    - **api/**
      - `__init__.py`
      - **routes/**
        - `search.py` — Search

- **data/**
  - **raw/** — Source data
    - `media.csv`
    - `places.csv`
    - `reviews.csv`
    - **AGREEMENTS + LICENSE/**
      - `data-license.md`
      - `usage-agreement.md`
  - **processed/** — Merged data
    - `merged.csv`
  - **indices/** — FAISS indices
    - `metadata.index`
    - `reviews.index`

- **frontend/**
  - `index.html`
  - **static/**
    - `api.js`
    - `ui.js`
    - `style.css`

- **scripts/**
  - `build_index.py` — Build merged CSV and FAISS indices
  - `run_server.sh` — Launch FastAPI server

- `.gitignore`
- `README.md`

## Setup Instructions

### 1. Backend Setup

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Data/Training
Make sure
`data/processed/merged.csv`
`data/indices/*.index`
exists, if not, run
```bash
python3 ../scripts/build_index.py
```

## Backend
From the Project root, run
```bash
./scripts/run_server.sh
# chmod +x ./scripts/run_server.sh
# ./scripts/run_server.sh
```
Start the FastAPI server (`uvicorn app.main:app`) on `http://127.0.0.1:8000`

## Frontend(Static)

```bash
cd frontend
python3 -m http.server 5500
```
Visit `http://127.0.0.1:5500` in browser.

## Flow
1. User opens frontend in a browser (`http://127.0.0.1:5500`)
2. User types query and hit "Search"
3. Frontend sends HTTP request to backend (`http://127.0.0.1:8000/search?q=...`)
4. Backend:
    - Encodes the query using the model
    - Searches the FAISS index
    - Returns the 5 closest matches
5. Frontend display the matching places

