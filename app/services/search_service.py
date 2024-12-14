import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import torch
import numpy as np
import os
from typing import List, Dict
from dotenv import load_dotenv
from pymongo import MongoClient
from bson import ObjectId

load_dotenv()
access_token = os.getenv("MY_ACCESS_TOKEN_HUGGINGFACE")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
reranker_model = AutoModelForSequenceClassification.from_pretrained(
    os.getenv("MY_RERANK_MODEL"),
    token=access_token
).to(device)
reranker_tokenizer = AutoTokenizer.from_pretrained(os.getenv("MY_RERANK_MODEL"))

client = MongoClient(os.getenv("MY_URI_MONGODB"))
db = client['mydatabase']

example_prompt = f"""Given a customer's 'Previous Purchases', rerank a list of 'Recommended Products' from most to least relevant to the customer's preferences. Only recommend products from the latest 'Recommended Products' section. The relevance should be determined by considering the types and themes of products the customer has bought before.

Example 1:
- User Input:
Previous Purchases:
1. BLUE CALCULATOR RULER
2. DOORMAT TOPIARY
3. PARTY BUNTING
Recommended Products:
1. CRYSTAL FROG PHONE CHARM
2. PINK CRYSTAL SKULL PHONE CHARM
3. BLUE LEAVES AND BEADS PHONE CHARM

- Model Output:
Reranked Recommendations:
1. BLUE LEAVES AND BEADS PHONE CHARM
2. CRYSTAL FROG PHONE CHARM
3. PINK CRYSTAL SKULL PHONE CHARM

Example 2:
- User Input:
Previous Purchases:
1. PANTRY HOOK SPATULA
2. BIRDCAGE DECORATION TEALIGHT HOLDER
3. REGENCY TEA PLATE PINK
Recommended Products:
1. SWEETHEART CAKESTAND 3 TIER
2. CAKESTAND, 3 TIER, LOVEHEART
3. REGENCY CAKESTAND 3 TIER

- Model Output:
Reranked Recommendations:
1. REGENCY CAKESTAND 3 TIER
2. SWEETHEART CAKESTAND 3 TIER
3. CAKESTAND, 3 TIER, LOVEHEART

"""

def cosine_similarity(query_vector, product_vector):
    query_vector = np.array(query_vector)
    product_vector = np.array(product_vector)
    return np.dot(query_vector, product_vector) / (np.linalg.norm(query_vector) * np.linalg.norm(product_vector))

def extract_initial_candidates(result: List[Dict], query_vector: np.ndarray) -> List[Dict]:
    candidates = [
        {
            "_id": str(doc["_id"]),
            "name": doc["name"],
            "sizes": doc["sizes"],
            "category": doc["category"],
            "color": doc["color"],
            "images": doc["images"],
            "total_stock": doc["total_stock"],
            "review_count": doc["review_count"],
            "rating_count": doc["rating_count"],
            "description": doc["description"],
            "avg_rating": doc["avg_rating"],
            "price": doc["price"],
            "description_embedding": doc["description_embedding"],
            "cosine_similarity": cosine_similarity(query_vector, doc["description_embedding"])
        }
        for doc in result
    ]
    return candidates

def sort_candidates_by_similarity(candidates: List[Dict], top_k: int) -> List[Dict]:
    return sorted(candidates, key=lambda x: x["cosine_similarity"], reverse=True)[:top_k]

def sort_candidates_by_rerank_scores(candidates: List[Dict], top_k: int) -> List[Dict]:
    return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_k]

def search_similar_products(user_embedding, top_n=100):
    # Convert the user embedding to a list for MongoDB compatibility
    query_embedding = user_embedding.tolist()
    # Perform the vector search in MongoDB using the aggregation pipeline
    results = db.products.aggregate([
        {
            "$vectorSearch": {
                "index": "vector_index",  # The name of your vector index
                "path": "description_embedding",  # The field storing the product embeddings
                "queryVector": query_embedding,  # The user query vector
                "numCandidates": 100,  # Number of candidates to search from
                "limit": top_n  # Limit the results to top N
            }
        }
    ])
    return results


def search_similar_products_none_tolist(user_embedding, top_n=100):
    # Convert the user embedding to a list for MongoDB compatibility
    query_embedding = user_embedding
    # Perform the vector search in MongoDB using the aggregation pipeline
    cursor = db.products.aggregate([
        {
            "$vectorSearch": {
                "index": "vector_index",  # The name of your vector index
                "path": "description_embedding",  # The field storing the product embeddings
                "queryVector": query_embedding,  # The user query vector
                "numCandidates": 100,  # Number of candidates to search from
                "limit": top_n  # Limit the results to top N
            }
        }
    ])
    # Convert the cursor to a list
    results = list(cursor)
    for result in results:
        if "_id" in result and isinstance(result["_id"], ObjectId):
            result["_id"] = str(result["_id"])

    return results

def rerank(query, sorted_candidates):
    """
    Rerank the sorted candidates based on their relevance to the query and return the top 6 results.

    Args:
        query (str): The user query.
        sorted_candidates (list): List of product dictionaries sorted by initial cosine similarity.

    Returns:
        list: Top 6 reranked product dictionaries with an added 'reason' field.
    """
    # Prepare the input prompt for reranking
    product_descriptions = [
        f"{i + 1}. {product['name']} (ID: {product['_id']})"
        for i, product in enumerate(sorted_candidates)
    ]
    queries = [query]*len(product_descriptions)
    features = reranker_tokenizer(queries, product_descriptions,  padding=True, truncation=True, return_tensors="pt").to(device)
    # Tokenize the input prompt
    with torch.no_grad():
            scores = reranker_model(**features).logits
            values, indices = torch.sum(scores, dim=1).sort()
    return [sorted_candidates[indices[0]],sorted_candidates[indices[1]],sorted_candidates[indices[2]]]