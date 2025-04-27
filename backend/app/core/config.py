from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    BASE_DIR: Path = Path(__file__).resolve().parents[3]
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DIR: Path = DATA_DIR / "processed"
    INDICES_DIR: Path = DATA_DIR / "indices"

    RAW_MEDIA_PATH: Path = RAW_DATA_DIR / "media.csv"
    RAW_PLACES_PATH: Path = RAW_DATA_DIR / "places.csv"
    RAW_REVIEWS_PATH: Path = RAW_DATA_DIR / "reviews.csv"
    PROCESSED_CSV_PATH: Path = PROCESSED_DIR / "merged.csv"

    METADATA_INDEX_PATH: Path = INDICES_DIR / "metadata.index"
    REVIEWS_INDEX_PATH: Path = INDICES_DIR / "reviews.index"

    MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

settings = Settings()
