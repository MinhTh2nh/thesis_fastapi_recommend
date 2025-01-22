import aiohttp
import pandas as pd
from fastapi import APIRouter, HTTPException, logger
from app.services.preprocessing_service import clean_file
from sentence_transformers import SentenceTransformer
from fastapi.responses import JSONResponse
from pymongo import MongoClient
from dotenv import load_dotenv
from typing import List
import numpy as np
from fastapi import File, UploadFile, Form
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
        url = "https://thesis-be.onrender.com/api/products/chunking"

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
