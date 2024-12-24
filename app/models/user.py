from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from pydantic import conint

class QueryRequest(BaseModel):
    query: str
    session_context: list = []  
    # num_candidates: int = 30  
    # limit: int = 30           
    # top_k: conint(ge=1) = 30  

class SearchResponse(BaseModel):
    user_query: str
    session_context: List[str]
    refined_products: List[dict]