import os
import torch
import hnswlib
import logging
import numpy as np
from fastapi import APIRouter
from dotenv import load_dotenv
from pymongo import MongoClient
from huggingface_hub import login
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from app.services.search_service import search_similar_products
from accelerate import init_empty_weights, infer_auto_device_map
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

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


# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# LLM and Tokenizer Setup
def load_llm_model():
    try:
        model_name = os.getenv("MY_RECOMMENDED_MODEL")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        with init_empty_weights():
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
            )
            if device != "cpu":
                device_map = infer_auto_device_map(
                    model,
                    no_split_module_classes=["BloomBlock"],
                    max_memory={0: "24GB", "cpu": "48GB"},
                )
            model.tie_weights()

        return tokenizer, model
    except Exception as e:
        logger.error(f"Error loading LLM model: {e}")
        raise

tokenizer_llm, model_llm = load_llm_model()
model_embedding = SentenceTransformer(os.getenv("MY_EMBEDDING_MODEL"))

logger.info(f"Using device: {device}")

# LLM Pipeline
def create_llm_pipeline(tokenizer, model):
    try:
        return pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=100,
            top_k=50,
            temperature=0.1,
            device=0 if device == "cuda" else -1,
        )
    except Exception as e:
        logger.error(f"Failed to create LLM pipeline: {e}")
        raise

model_pipeline = create_llm_pipeline(tokenizer_llm, model_llm)
llm = HuggingFacePipeline(pipeline=model_pipeline)

# Prompt Template
def get_prompt_template():
    template = (
        "You are an AI assistant on an e-commerce website. "
        "Your job is to recommend the top 5 products tailored to the user's purchase history and shopping cart.\n\n"
        "### Context:{context} \n\n"
        "### Human: {question}\n\n"
        "### Assistant:"
    )
    return PromptTemplate(template=template, input_variables=["context", "question"])

# User Preferences Extraction
def extract_user_preferences(user_data):
    # Default to empty lists or zero ranges if no data is available
    categories = user_data.get("search_history", [])
    orders = user_data.get("order_products", [])
    prices = [item.get("price", 0) for item in orders] if orders else []
    
    # Return empty or default values for preferences if data is missing
    return {
        "categories": categories,
        "price_range": (min(prices, default=0), max(prices, default=0)),
    }

# Calculate Cosine Similarity
def calculate_similarity(user_embedding, product_embedding):
    try:
        user_norm = user_embedding / np.linalg.norm(user_embedding)
        product_norm = product_embedding / np.linalg.norm(product_embedding)
        return np.dot(user_norm, product_norm)
    except Exception as e:
        logger.error(f"Error calculating similarity: {e}")
        return 0.0


# Generate User Embedding
def generate_user_embedding(preferences):
    return model_embedding.encode(preferences, convert_to_tensor=True)

# Recommend Products
def recommend_products(user_data, products):
    if not user_data or not isinstance(user_data, dict):
        logger.error("Invalid user data: Missing or malformed user_data.")
        return []  # Return an empty list if user data is invalid

    preferences = extract_user_preferences(user_data)
    user_embedding = generate_user_embedding(preferences)
    recommendations = []

    for product in products:
        product_embedding = np.array(product["description_embedding"], dtype=np.float32)
        similarity = calculate_similarity(user_embedding, product_embedding)
        recommendations.append({
            **product, 
            "score": float(similarity),  # Convert score to native float
            "product_id": str(product.get("product_id", ""))  # Ensure product ID is string
        })

    return sorted(recommendations, key=lambda x: x["score"], reverse=True)


# Load RAG Pipeline
def create_rag_pipeline(llm, retriever, prompt):
    try:
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True,
        )
    except Exception as e:
        logger.error(f"Error creating RAG pipeline: {e}")
        raise

# RAG Handler
class RAGPipelineHandler:
    def __init__(self, retriever=None):
        self.retriever = retriever or search_similar_products
        self.llm = llm
        self.prompt = get_prompt_template()

    def rag(self, user_data, similar_products):
        try:
            refined_recommendations = recommend_products(user_data, similar_products)
            # Convert results to native Python types
            for recommendation in refined_recommendations:
                recommendation["score"] = float(recommendation["score"])  # Ensure float type
                recommendation["product_id"] = str(recommendation["product_id"])  # Ensure ID is string
            return refined_recommendations[:5]
        except Exception as e:
            logger.error(f"Error refining recommendations: {e}")
            raise

