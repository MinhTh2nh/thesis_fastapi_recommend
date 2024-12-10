from app.models.product import Product
from pydantic import BaseModel
from typing import Any, List, Dict, Optional
from pydantic import BaseModel
from fastapi import File, UploadFile

class EmbeddingRequest(BaseModel):
    model: str
    json_list: List[Dict]  # Input for embedding server

class ChunkingRequest(BaseModel):
    files: List[UploadFile]
    model: str

# Define Response Models
class ChunkingResponse(BaseModel):
    chunking_products: List[Product]  # List of products in JSON format

class EmbeddingResponse(BaseModel):
    _id: str
    name: str
    product_id: int
    category: str
    price: str
    color: str
    sizes: List[Dict[str, Any]]
    description: List[str]
    images: List[str]
    total_stock: int
    sold_count: int
    review_count: int
    rating_count: int
    avg_rating: float
    description_embedding: List[float]  # Array of 384 embedding values
    reviews: List[Dict[str, Any]]
    total_rating: int