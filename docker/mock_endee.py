#!/usr/bin/env python3
"""
Mock Endee server for testing purposes.
Provides basic vector storage and search endpoints.
"""

import json
import uuid
from datetime import datetime
from typing import Dict, List, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Mock Endee Server", version="1.0.0")

# In-memory storage for vectors
vector_store: Dict[str, Dict[str, Any]] = {}

class VectorAddRequest(BaseModel):
    id: str
    vector: List[float]
    metadata: Dict[str, Any]

class VectorSearchRequest(BaseModel):
    vector: List[float]
    top_k: int = 5

class VectorSearchResult(BaseModel):
    id: str
    score: float
    metadata: Dict[str, Any]

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/vectors/add")
async def add_vector(request: VectorAddRequest):
    """Add a vector to the database."""
    try:
        # Validate vector dimension
        if len(request.vector) != 384:
            raise HTTPException(
                status_code=400, 
                detail=f"Expected 384 dimensions, got {len(request.vector)}"
            )
        
        # Store vector
        vector_store[request.id] = {
            "vector": request.vector,
            "metadata": request.metadata,
            "created_at": datetime.now().isoformat()
        }
        
        return {
            "status": "success",
            "id": request.id,
            "dimension": len(request.vector)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/vectors/search")
async def search_vectors(request: VectorSearchRequest):
    """Search for similar vectors."""
    try:
        if len(request.vector) != 384:
            raise HTTPException(
                status_code=400,
                detail=f"Expected 384 dimensions, got {len(request.vector)}"
            )
        
        # Simple cosine similarity calculation
        results = []
        for doc_id, doc_data in vector_store.items():
            # Calculate dot product (since vectors are normalized)
            score = sum(a * b for a, b in zip(request.vector, doc_data["vector"]))
            results.append({
                "id": doc_id,
                "score": float(score),
                "metadata": doc_data["metadata"]
            })
        
        # Sort by score descending and return top_k
        results.sort(key=lambda x: x["score"], reverse=True)
        return {"results": results[:request.top_k]}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/vectors/count")
async def get_vector_count():
    """Get the number of vectors stored."""
    return {"count": len(vector_store)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)