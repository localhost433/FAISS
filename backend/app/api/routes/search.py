# app/api/routes/search.py

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from app.core.hybrid_utils import run_hybrid_search

router = APIRouter()

@router.get("/search")
async def hybrid_search(q: str = Query(..., description="Search query"), top_k: int = 5):
    try:
        results = await run_hybrid_search(q, top_k)
        return {"results": results}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Search failed: {str(e)}"}
        )
