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

def get_user_embeddings(orders, cart_items, search_history):
    """
    Generate embeddings for a user's interactions based on their orders, cart items, and search history.

    Args:
        orders (list): A list of orders, each containing a dictionary with product details.
        cart_items (list): A list of items in the user's cart, each containing a dictionary with product details.
        search_history (list): A list of product identifiers or descriptions from the user's search history.

    Returns:
        torch.Tensor or None: A tensor of product embeddings or None if no valid products are found.
    """
    all_products = []  # List to store product identifiers or descriptions

    # Extract product IDs from orders
    if orders and isinstance(orders, list):
        for order in orders:
            if isinstance(order, dict):
                for item in order.get('items', []):  # Use .get() to avoid KeyError
                    if isinstance(item, dict):
                        product_id = item.get('productId')  # Safely retrieve product ID
                        if product_id:
                            all_products.append(str(product_id))

    # Extract product IDs from cart items
    if cart_items and isinstance(cart_items, list):
        for cart_item in cart_items:
            if isinstance(cart_item, dict):
                product_id = cart_item.get('productId')
                if product_id:
                    all_products.append(str(product_id))

    # Add product identifiers or descriptions from search history
    if search_history and isinstance(search_history, list):
        all_products.extend(str(product) for product in search_history if isinstance(product, (str, int)))

    # Handle case where no valid product data is found
    if not all_products:
        return None  # Return None or a default value (e.g., an empty tensor)

    # Generate embeddings for all products
    try:
        embeddings = model_embedding.encode(all_products, convert_to_tensor=True)  # Assuming `model` is preloaded
        return embeddings
    except Exception as e:
        logging.error(f"Failed to generate embeddings: {e}")
        return None

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
        # Aggregate user embeddings to a single vector
        if len(user_embedding.shape) > 1:
            user_embedding = torch.mean(user_embedding, dim=0)  # Reduce to 1D
        # Convert tensor to list of floats
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

def generate_recommendations(user_data, recommended_products):
    """
    Generate reranked product recommendations based on user data.

    Args:
        user_data (dict): The user's interaction history (orders, cart items, search history).
        recommended_products (list): The initial list of recommended products.

    Returns:
        list: Reranked list of product recommendations.
    """
    try:
        # Extract and format user history data
        order_history = "\n".join([f"{entry['product_name']} (Size: {entry['size']}, Price: {entry['price']})"
                                   for entry in user_data.get('order_products', [])])
        
        cart_history = "\n".join([f"{entry['product_name']} (Size: {entry['size']}, Price: {entry['price']})"
                                  for entry in user_data.get('cart_products', [])])
        
        search_history = "\n".join([f"{entry}" for entry in user_data.get('search_history', [])])

        # Combine history into a single string
        user_context = f"Order History:\n{order_history if order_history else 'No orders yet.'}\n" \
                       f"Cart Items:\n{cart_history if cart_history else 'No items in the cart.'}\n" \
                       f"Search History:\n{search_history if search_history else 'No search history.'}"

        # Rerank the products
        reranked_products = rerank_products(user_context, recommended_products)

        return reranked_products

    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        return []






