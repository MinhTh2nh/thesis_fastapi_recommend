import os
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from pymongo import MongoClient
from dotenv import load_dotenv
from huggingface_hub import login
from accelerate import init_empty_weights, infer_auto_device_map
from app.services.search_service import search_similar_products
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from fastapi import APIRouter
from langchain_huggingface import HuggingFacePipeline

# FastAPI Router
router = APIRouter()

# Log in to Hugging Face using the token
login(os.getenv("MY_ACCESS_TOKEN_HUGGINGFACE"))

# Load environment variables
load_dotenv()

# MongoDB connection setup
try:
    client = MongoClient(os.getenv("MY_URI_MONGODB"))
    db = client["mydatabase"]
except Exception as e:
    raise Exception(f"Failed to connect to MongoDB: {str(e)}")

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load LLM tokenizer and model
tokenizer_llm = AutoTokenizer.from_pretrained(os.getenv("MY_RECOMMENDED_MODEL"))

with init_empty_weights():
    model_llm = AutoModelForCausalLM.from_pretrained(
        os.getenv("MY_RECOMMENDED_MODEL"),
        trust_remote_code=True,
        top_k=10,
        top_p=0.95,
        temperature=0.4,
    )

device_map = infer_auto_device_map(
    model_llm,
    no_split_module_classes=["BloomBlock"],
    max_memory={0: "24GB", "cpu": "48GB"}
)

model_pipeline = pipeline(
    "text-generation",
    model=model_llm,
    tokenizer=tokenizer_llm,
    max_new_tokens=100, 
    top_k=50, 
    temperature=0.1,
    device=0 if torch.cuda.is_available() else -1
)

llm = HuggingFacePipeline(pipeline=model_pipeline)

# Prompt Template for RAG Pipeline
def load_prompt_template():
    query_template = (
        "You are an AI assistant on an e-commerce website. "
        "Your job is to recommend the top 5 products tailored "
        "to the user's purchase history and shopping cart.\n\n"
        "### Context:{context} \n\n"
        "### Human: {question}\n\n"
        "### Assistant:"
    )
    return PromptTemplate(template=query_template, input_variables=["context", "question"])

# Extract user preferences
def extract_user_preferences(user_data):
    """
    Extracts user preferences including categories and price range.
    """
    search_history = user_data.get("search_history", [])
    orders = user_data.get("order_products", [])
    
    categories = [term for term in search_history if term]
    prices = [item["price"] for item in orders] if orders else []
    price_range = (min(prices), max(prices)) if prices else None
    
    return {"categories": categories, "price_range": price_range}

# Calculate similarity using Llama model
def calculate_similarity_llama(user_preferences, product):
    """
    Generates a similarity score for a product based on user preferences.
    """
    print(f"User Preferences: {user_preferences}")
    print(f"Product: {product}")
    
    if not user_preferences['categories'] or not user_preferences['price_range']:
        print("No valid categories or price range found.")
        return 0.0

    prompt = f"""
    User is interested in categories {user_preferences['categories']} and price range {user_preferences['price_range']}.
    Compare this preference with the product: {product['name']} - {product['description']}.
    Provide a similarity score between 0 and 1.
    """
    inputs = tokenizer_llm(prompt, return_tensors="pt").to(device)
    outputs = model_llm.generate(
        **inputs,
        max_new_tokens=50,
        pad_token_id=tokenizer_llm.eos_token_id
    )
    response = tokenizer_llm.decode(outputs[0], skip_special_tokens=True)
    try:
        score = float(response.strip().split()[-1])  # Extracts score
    except ValueError:
        score = 0.0
    print(f"Generated Score: {score}")
    return score

# Refine product recommendations using Llama
def recommend_products_llm(user_data, products):
    """
    Recommends products by calculating similarity scores and sorting them.
    """
    user_preferences = extract_user_preferences(user_data)
    recommendations = []
    for product in products:
        similarity_score = calculate_similarity_llama(user_preferences, product)
        product_with_score = product.copy()
        product_with_score["score"] = similarity_score
        recommendations.append(product_with_score)
    
    # Sort the recommendations based on the similarity score
    return sorted(recommendations, key=lambda x: x["score"], reverse=True)

# Load RAG pipeline
def load_prompt_template():
    query_template = (
        "You are an AI assistant on an e-commerce website. "
        "Your job is to recommend the top 5 products tailored "
        "to the user's purchase history and shopping cart.\n\n"
        "### Context:{context} \n\n"
        "### Human: {question}\n\n"
        "### Assistant:"
    )
    return PromptTemplate(template=query_template, input_variables=["context", "question"])

# Load the RAG pipeline
def load_rag_pipeline(llm, retriever, prompt):
    """
    Initializes the RAG pipeline with the specified LLM and retriever.
    """
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

# Usage in RAG handler
class RAGPipelineHandler:
    def __init__(self, embeddings=None, retriever=None):
        self.embeddings = embeddings
        self.retriever = retriever if retriever else search_similar_products  # Default to search_similar_products
        self.llm = llm
        self.prompt = load_prompt_template()

    def rag(self, user_data, similar_products):
        """
        Refines recommendations using the RAG pipeline.
        """
        try:
            context = "\n".join([f"{p['name']}: {p['description']}" for p in similar_products])
            user_preferences = extract_user_preferences(user_data)
            question = (
                f"Based on the user's interests in categories {user_preferences['categories']} "
                f"and price range {user_preferences['price_range']}, recommend the top 5 products."
            )
            
            # Generate refined recommendations using the LLM
            refined_recommendations = recommend_products_llm(user_data, similar_products)
            return sorted(refined_recommendations, key=lambda x: x["score"], reverse=True)[:5]
        except Exception as e:
            raise Exception(f"Error in refining recommendations: {str(e)}")