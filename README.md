# FAISS Search local implementation

## Structure
The project should look like this:

       FAISS/
       ├── README.md
       ├── backend/
       │   ├── __init__.py
       │   ├── data/
       │   │   ├── media.csv
       │   │   ├── merged.csv
       │   │   ├── places.csv
       │   │   ├── processed.csv
       │   │   └── reviews.csv
       │   ├── faiss_indices/
       │   │   └── index_file.index
       │   ├── main.py
       │   ├── requirements.txt
       │   └── training.py
       ├── index.html
       └── src/
           ├── api.js
           ├── main.js
           └── ui.js


## Setup
       cd backend
       python3 -m venv venv
       source venv/bin/activate
       pip install -r requirements.txt
Will take a while and might be differnet on Windows that run 
`venv\Scripts\activate`
instead of
`source venv/bin/activate`

## Data/Training
Make sure
`backend/data/merged.csv`
`backend/faiss_indices/index_file.index`
exists, if not, run

       python3 training.py
(Training script that builds the index and metadata from the dataset.)

## Backend
In `backend/` run

       uvicorn main:app --reload --port 8000
Server will be running at: `http://127.0.0.1:8000`

## Frontend
In a new terminal tab, run

       cd path/to/FAISS/
       python3 -m http.server 5500
Visit `http://127.0.0.1:5500/index.html` in your browser to use the app.

## Troubleshooting Guide
| Problem                                                                                                                     | How to Fix                                                                                    |
|-----------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|
| `ModuleNotFoundError: No module named 'app'`                                                                                | Run `uvicorn main:app --reload --port 8000` inside the `backend/` directory.                  |
| `FileNotFoundError: ../data/merged.csv not found` or `RuntimeError: cannot open faiss_indexes/index_file.index for reading` | Run `python training.py`.                                                                     |
| `Numpy error`                                                                                                               | Downgrade numpy: `pip install numpy==1.26.4`.                                                 |
| Frontend not showing results                                                                                                | Confirm backend is live at `http://127.0.0.1:8000` and frontend is at `http://127.0.0.1:5500`.|

## Flowchart for local deployment
       [ User (browser) ]
              |
              |  (typing query)
              v
       [ Frontend Server (http://127.0.0.1:5500) ]
              |
              |  (fetch request: http://127.0.0.1:8000/search?q=...)
              v
       [ Backend Server (FastAPI at http://127.0.0.1:8000) ]
              |
              |  (query processing: search FAISS index)
              v
       [ FAISS index + CSV metadata files ]
