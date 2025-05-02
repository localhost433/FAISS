# app/core/faiss_io.py

from pathlib import Path
import faiss
import numpy as np

from app.core.config import settings

METADATA_INDEX_PATH = Path(__file__).resolve().parents[2] / "data/indices/metadata.index"
REVIEWS_INDEX_PATH = Path(__file__).resolve().parents[2] / "data/indices/reviews.index"
IMAGE_INDEX_PATH = Path(__file__).resolve().parents[2] / "data/indices/image.index"

def build_faiss_index(vectors: np.ndarray) -> faiss.Index:
    try:
        d = vectors.shape[1]
        index = faiss.IndexFlatL2(d)  # L2 distance (Euclidean)
        index.add(vectors)
        print(f"FAISS index built with {index.ntotal} vectors of dimension {d}.")
        return index
    except Exception as e:
        print(f"Error building FAISS index: {e}")
        raise

def save_faiss_index(index: faiss.Index, path: Path):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(path))
        print(f"FAISS index saved to {path}.")
    except Exception as e:
        print(f"Error saving FAISS index to {path}: {e}")
        raise

def load_faiss_index(path: Path) -> faiss.Index:
    if not isinstance(path, Path):
        raise TypeError(f"Expected a Path object, but got {type(path).__name__}")
    try:
        # Attempt to load the FAISS index from the specified path.
        # Note: If the file is invalid or corrupted, `faiss.read_index` may raise an exception.
        return faiss.read_index(str(path))
    except FileNotFoundError:
        raise FileNotFoundError(f"FAISS index not found at {path}")
    except Exception as e:
        print(f"Error loading FAISS index from {path}: {e}")
        raise
