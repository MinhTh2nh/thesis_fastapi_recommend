import os
import torch
import logging
from bson import ObjectId
from fastapi import APIRouter
from dotenv import load_dotenv
from pymongo import MongoClient
from huggingface_hub import login
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI Router
router = APIRouter()
# Load environment variables
load_dotenv()
# Hugging Face Authentication
try:
    login(os.getenv("MY_ACCESS_TOKEN_HUGGINGFACE"))
except Exception as e:
    logger.error(f"Failed to authenticate with Hugging Face: {e}")
    raise

# MongoDB Connection
def connect_to_mongodb():
    try:
        client = MongoClient(os.getenv("MY_URI_MONGODB"), serverSelectionTimeoutMS=5000)
        client.server_info()  # Verify connection
        return client["mydatabase"]
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise

db = connect_to_mongodb()
products_collection = db["products"]


# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# LLM and Tokenizer Setup
def load_llm_model():
    try:
        model_name = os.getenv("MY_RECOMMENDED_MODEL")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load the Flan-T5 model correctly based on the device
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="auto" if device == "cuda" else None,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )

        return tokenizer, model
    except Exception as e:
        logger.error(f"Error loading LLM model: {e}")
        raise

tokenizer_llm, model_llm = load_llm_model()
model_llm.to(device)
model_embedding = SentenceTransformer(os.getenv("MY_EMBEDDING_MODEL"))

logger.info(f"Using device: {device}")

def serialize_objectid(obj):
    """
    Recursively converts ObjectId fields to string in a dictionary or list.
    """
    if isinstance(obj, dict):
        return {key: serialize_objectid(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [serialize_objectid(item) for item in obj]
    elif isinstance(obj, ObjectId):
        return str(obj)
    else:
        return obj

def get_user_embeddings(preprocessed_user_data):
    query_embedding = model_embedding.encode(preprocessed_user_data, convert_to_tensor=True)
    return query_embedding

import torch

def search_similar_products_Rec(user_embedding, k=21):
    """
    Search for the top-k most similar products to the user's embedding.
    Args:
        user_embedding (torch.Tensor): The embedding representing the user's preferences (shape: [n, 384]).
        k (int): The number of similar products to retrieve.
    Returns:
        list: A list of dictionaries containing product IDs and product names of the most similar products.
    """
    try:
        query_embedding = user_embedding.cpu().numpy().tolist()
        
        # Perform the vector search in MongoDB
        results = db.products.aggregate([
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "description_embedding",
                    "queryVector": query_embedding,
                    "numCandidates": 100,
                    "limit": k
                }
            }
        ])
        
        # Convert the CommandCursor to a list
        results_list = list(results)
        return results_list

    except Exception as e:
        logger.error(f"Error in search_similar_products: {e}")
        return []

def get_popular_products(limit=21):
    """
    Retrieve popular or trending products from the database.
    Args:
        limit (int): The maximum number of products to retrieve.
    Returns:
        list: A list of popular or trending products.
    """
    popular_products = list(
        products_collection.find()
        .sort([("sold_count", -1), ("avg_rating", -1)])  # Sort by sold_count and avg_rating
        .limit(limit)
    )
    return [serialize_objectid(product) for product in popular_products]

    
def rerank_products(user_context, recommended_products):
    """
    Rerank the recommended products based on user context using BAAI/bge-reranker-v2-m3.

    Args:
        user_context (str): A string summarizing the user's interaction history.
        recommended_products (list): A list of product dictionaries with 'name' and '_id' fields.

    Returns:
        list: A list of reranked product recommendations.
    """
    try:
        # Prepare query-product pairs for scoring
        queries = [user_context] * len(recommended_products)
        product_details = [
            f'{product.get("name", "Unknown Product")} (Color: {product.get("color", "Unknown Color")}, Price: ${product.get("price", "Unknown Price")})'
            for product in recommended_products
        ]
        # Tokenize inputs for the reranker
        inputs = tokenizer_llm(
            queries, 
            product_details, 
            padding=True, 
            truncation=True, 
            return_tensors="pt", 
            max_length=512
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # Generate relevance scores
        with torch.no_grad():
            scores = model_llm(**inputs, return_dict=True).logits.view(-1).float()  # Ensure scores are 1D

        # Attach scores to the products
        for product, score in zip(recommended_products, scores):
            product["score"] = score.item()

        # Sort products by scores in descending order
        reranked_products = sorted(recommended_products, key=lambda x: x["score"], reverse=True)

        return reranked_products

    except Exception as e:
        logger.error(f"Error reranking products: {e}")
        return []






