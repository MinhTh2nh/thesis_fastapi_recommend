from pydantic import BaseModel
from typing import Any, List, Dict

class Product(BaseModel):
    _id: str
    product_id: int
    name: str
    category: str
    price: str
    color: str
    sizes: List[Dict[str, int]]
    description: Dict[str, str]
    images: List[str]
    total_stock: int
    sold_count: int
    review_count: int
    rating_count: int
    avg_rating: float
    reviews: List[Dict[str, Any]]
    total_rating: int

class RecommendRequest(BaseModel):
    user_id: str

class RecommendResponse(BaseModel):
    user_id: int
    recommended_products: List[Product]
