import traceback
from aiohttp import ClientSession
import aiohttp
import pandas as pd
from app.models.embedding import ChunkingRequest, EmbeddingRequest
from app.services.preprocessing_service import clean_file, generate_chunking_products, generate_embeddings
from fastapi import APIRouter, HTTPException, logger
from app.services.user_service import get_user_data, preprocess_user_data
from app.services.search_service import cosine_similarity, extract_initial_candidates, sort_candidates_by_similarity, rerank, search_similar_products, search_similar_products_none_tolist
from app.services.embedding_service import get_user_embeddings_context
from app.services.recommendation import RAGPipelineHandler, extract_user_preferences
from app.models.product import RecommendRequest
from app.models.user import QueryRequest
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
from typing import List
import os
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import numpy as np
from fastapi import FastAPI, File, UploadFile, Request, HTTPException, Form
import json

load_dotenv()

router = APIRouter()

model = SentenceTransformer(os.getenv("MY_EMBEDDING_MODEL"))
client = MongoClient(os.getenv("MY_URI_MONGODB"))
db = client['mydatabase']
products_collection = db["products"]

@router.post("/recommend")
async def recommend_products(request: RecommendRequest):
    try:
        # Fetch user data
        user, orders, cart_items, search_history = get_user_data(request.user_id)

        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        processed_user_data = preprocess_user_data(user, orders, cart_items, search_history)
        # Generate user embedding11
        session_context = get_user_embeddings_context(processed_user_data)
        query_vector = model.encode(session_context)

        # Ensure the vector is serializable
        query_vector_serializable = query_vector.tolist() 
        similar_products = search_similar_products_none_tolist(query_vector_serializable)
        rag_handler = RAGPipelineHandler(retriever=search_similar_products)  # Pass the retriever explicitly
        recommended_products = rag_handler.rag(processed_user_data, similar_products)
        return {
            "user_id": request.user_id,
            "similar_products": similar_products,
            "recommended_products": recommended_products,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/search")
# @router.post("/search")
async def search_products(request: QueryRequest):
    try:
        # Build full query from session context and the current query
        session_context = request.session_context + [request.query]
        full_query = " ".join(session_context)
        # Generate query vector using the embedding model
        query_vector = model.encode(full_query)
        # Fetch initial candidates from MongoDB using vector search
        result = search_similar_products(query_vector)
        # Collect initial candidates and their embeddings
        initial_candidates = extract_initial_candidates(result, query_vector)
        # Calculate cosine similarity between query and product embeddings
        for candidate in initial_candidates:
            candidate["cosine_similarity"] = cosine_similarity(query_vector, candidate["description_embedding"])
        # Sort candidates by cosine similarity and limit to top_k
        sorted_candidates = sort_candidates_by_similarity(initial_candidates, request.top_k)
        sorted_candidates_without_embedding = [
            {**{key: value for key, value in candidate.items() if key != 'description_embedding'}, 'rank': index + 1}
            for index, candidate in enumerate(sorted_candidates)
        ]
        # Rerank candidates using the rerank service
        reranked_candidates = rerank(request.query, sorted_candidates)
        # Build final response
        return JSONResponse(
            status_code=200,
            content={
                "user_query": request.query,
                "session_contex": session_context,
                "refined_products": sorted_candidates_without_embedding,
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
        
        # Generate chunking products from the combined data
        products = generate_chunking_products(cleaned_data_combined)

        # Prepare data for POST request
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
