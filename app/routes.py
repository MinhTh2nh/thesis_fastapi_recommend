from app.models.embedding import ChunkingRequest, EmbeddingRequest
from app.services.preprocessing_service import clean_file, generate_chunking_products, generate_embeddings
from fastapi import APIRouter, HTTPException
from app.services.user_service import get_user_data, preprocess_user_data
from app.services.search_service import get_query_embedding, search_similar_products, rerank_products
# from app.services.embedding_service import get_user_embeddings_context
# from app.services.recommendation import RAGPipelineHandler, 
from app.services.personalized_production import get_user_embeddings,search_similar_products, generate_recommendations, serialize_objectid
from app.models.product import RecommendRequest
from app.models.user import QueryRequest
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
from typing import List
import os
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import numpy as np
import logging

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

router = APIRouter()

model = SentenceTransformer(os.getenv("MY_EMBEDDING_MODEL"))
client = MongoClient(os.getenv("MY_URI_MONGODB"))
db = client['mydatabase']
products_collection = db["products"]


@router.post("/recommend")
async def recommend_products(request: RecommendRequest):
    try:
        # Step 1: Fetch and validate user data
        orders, cart_items, search_history = get_user_data(request.user_id)
        logging.info(f"User data fetched successfully for user ID: {request.user_id}")
        processed_user_data = preprocess_user_data(orders, cart_items, search_history)

        # Step 3: Generate user embedding
        user_embedding = get_user_embeddings(
            processed_user_data["order_products"],
            processed_user_data["cart_products"],
            processed_user_data["search_history"],
        )
        if user_embedding is None:
            raise HTTPException(status_code=500, detail="Failed to generate user embedding")

        # Step 4: Retrieve similar products
        similar_products = search_similar_products(user_embedding)
        if not similar_products:
            return JSONResponse(
                status_code=404,
                content={"message": "No similar products found for the user."},
            )
        logging.info(f"Found {len(similar_products)} similar products")

        # **Ensure serialization of MongoDB ObjectId fields**
        serialized_products = [serialize_objectid(product) for product in similar_products]

        # Step 5: Generate LLM response for recommendations
        response = generate_recommendations(processed_user_data, serialized_products)
        if not response:
            raise HTTPException(status_code=500, detail="Failed to generate LLM response")

        # Step 6: Return serialized response
        return JSONResponse(
            status_code=200,
            content={
                "user_id": request.user_id,
                "recommendations": serialized_products,
                "message": response,
            },
        )

    except HTTPException as http_err:
        logging.error(f"HTTP error in /recommend API: {http_err.detail}")
        raise http_err
    except Exception as e:
        logging.error(f"Unexpected error in /recommend API: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred")


@router.post("/search")
# @router.post("/search")
async def search_products(request: QueryRequest):
    try:
        # Build full query from session context and the current query
        session_context = request.session_context + [request.query]
        # full_query = " ".join(session_context)
        # Generate query vector using the embedding model
        query_vector = get_query_embedding(request)  # Pass the whole request object
        # Fetch initial candidates from MongoDB using vector search
        results = search_similar_products(query_vector)
        serialized_results = serialize_objectid(results)
        # Rerank candidates using the rerank service
        reranked_candidates = rerank_products(request.query, serialized_results)
        # Build final response
        return JSONResponse(
            status_code=200,
            content={
                "user_query": request.query,
                "session_contex": session_context,
                "refined_products": serialized_results,
                "rank_product_llm": reranked_candidates,
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@router.post("/chunking")
async def process_chunking(request: ChunkingRequest):
    try:
        # Clean the uploaded file
        cleaned_data = clean_file(request.files)
        products = generate_chunking_products(cleaned_data)
        return JSONResponse(
            status_code=200,
            content={
                 "chunking_products": products,
                 "model": request.model 
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    
@router.post("/embedding")
async def process_embedding(request: EmbeddingRequest):
    try:
        embedding_results = generate_embeddings(request.json_list)
        return JSONResponse(
            status_code=200,
            content={
                 "embedding_results": embedding_results,
                 "model": request.model 
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
