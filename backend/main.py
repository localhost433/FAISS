from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import faiss
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os

# Initialize app
app = FastAPI()

# Allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


faiss_index = faiss.read_index("faiss_indices/index_file.index")
metadata_path = os.path.join(os.path.dirname(__file__), "./data/merged.csv")
metadata_df = pd.read_csv(metadata_path)

# Load your embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

def search_faiss(query, top_k=5):
    query_vec = model.encode([query], normalize_embeddings=True)
    query_vec = np.array(query_vec).astype('float32')
    
    distances, indices = faiss_index.search(query_vec, top_k)
    results = []

    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue  # Skip if no valid result
        result = metadata_df.iloc[idx].to_dict()
        result['distance'] = float(dist)
        results.append(result)
    
    return results

@app.get("/search")
async def search(q: str = Query(..., description="Search query")):
    results = search_faiss(q)
    return {"results": results}
