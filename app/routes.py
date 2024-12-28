import traceback
from aiohttp import ClientSession
import aiohttp
import pandas as pd
from app.models.embedding import ChunkingRequest, EmbeddingRequest
from fastapi import APIRouter, HTTPException, logger
from app.services.personalized_production import get_user_embeddings,search_similar_products_Rec, serialize_objectid, get_popular_products
from app.services.preprocessing_service import clean_file, generate_embeddings
from app.services.search_service import get_query_embedding, search_similar_products, rerank_products
from app.services.user_service import get_user_data, preprocess_user_data
# from app.services.embedding_service import get_user_embeddings_context
from app.models.embedding import ChunkingRequest, EmbeddingRequest
from sentence_transformers import SentenceTransformer
from app.models.product import RecommendRequest
from fastapi.responses import JSONResponse
from app.models.user import QueryRequest
from pymongo import MongoClient
from dotenv import load_dotenv
from typing import List
import numpy as np
from fastapi import FastAPI, File, UploadFile, Request, Form
import json
import logging
import os

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
                # Check if user_id is provided in the request
        if not request.user_id:
            logging.warning("No user_id provided, recommending popular products.")
            
            # Recommend popular or trending products if no user_id is available
            popular_products = get_popular_products()
            return JSONResponse(
                status_code=200,
                content={
                    "message": "No user_id provided, recommending popular products.",
                    "recommendations": popular_products,
                },
            )
        # Step 1: Fetch and validate user data
        orders, cart_items, search_history = get_user_data(request.user_id)
        logging.info(f"User data fetched successfully for user ID: {request.user_id}")
        # Check if user data is empty or insufficient
        if (not orders or len(orders) == 0) and (not cart_items or len(cart_items) == 0) and (not search_history or len(search_history) == 0):
            logging.warning(f"Cold start detected for user ID: {request.user_id}")
            
            # Recommend popular or trending products for cold start users
            popular_products = get_popular_products()
            return JSONResponse(
                status_code=200,
                content={
                    "user_id": request.user_id,
                    "recommendations": popular_products,
                },
            )
        
        # Step 2: Process user data
        processed_user_data = preprocess_user_data(orders, cart_items, search_history)
        # Step 3: Generate user embedding
        user_embedding = get_user_embeddings(processed_user_data)
        if user_embedding is None:
            raise HTTPException(status_code=500, detail="Failed to generate user embedding")

        # Step 4: Retrieve similar products
        similar_products = search_similar_products_Rec(user_embedding)
        if not similar_products:
            return JSONResponse(
                status_code=404,
                content={"message": "No similar products found for the user."},
            )
        logging.info(f"Found {len(similar_products)} similar products")

        # Ensure serialization of MongoDB ObjectId fields
        serialized_products = [serialize_objectid(product) for product in similar_products]

        # Step 5: Generate LLM response for recommendations
        response = rerank_products(processed_user_data, serialized_products)
        if not response:
            raise HTTPException(status_code=500, detail="Failed to generate LLM response")

        # Step 6: Return serialized response
        return JSONResponse(
            status_code=200,
            content={
                "user_id": request.user_id,
                "recommendations": response,
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
                # "refined_products": serialized_results,
                "rank_product_llm": reranked_candidates,
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@router.post("/chunking")
async def process_chunking(
    files: List[UploadFile] = File(...),
    file_type: str = Form(...),
):
    try:
        # Process each file and clean data
        cleaned_data_list = [await clean_file(file) for file in files]
        
        # Combine all cleaned data into one DataFrame
        cleaned_data_combined = pd.concat(cleaned_data_list, ignore_index=True)
        products = cleaned_data_combined.to_dict(orient="records")

        # # Prepare data for POST request
        url = "http://localhost:3001/api/products/chunking"

        payload = {
            "chunking_list": products,
            "file_name": [file.filename for file in files],
            "file_type": file_type,
        }
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, json=payload) as response:
                    # Handle external response
                    if response.status == 200:
                        external_response = await response.json()
                        if external_response.get('status') == 'success':
                            data = external_response.get('data', {})

                            return JSONResponse(
                                status_code=200,
                                content={"data": data},
                            )
                        else:
                            return JSONResponse(
                                status_code=400,
                                content={"error": "Failed to process chunking data", "details": external_response},
                            )
                    else:
                        return JSONResponse(
                            status_code=response.status,
                            content={"error": f"Unexpected status code: {response.status}"},
                        )
            except Exception as e:
                logger.error('Error communicating with Node.js server:', exc_info=True)
                return JSONResponse(
                    content={"error": "An unexpected error occurred during external request", "details": str(e)},
                    status_code=500,
                )
            
        # Return JSON response
        return JSONResponse(
            status_code=200,
            content={
                "chunking_products": products,
                "file_names": [file.filename for file in files],
                "file_type": file_type,
                "payload": payload,
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
