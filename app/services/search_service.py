import re
import os
import nltk
import torch
import spacy
import logging
from fastapi import APIRouter
from dotenv import load_dotenv
from pymongo import MongoClient
from nltk.corpus import stopwords
from huggingface_hub import login
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from app.models.user import QueryRequest
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
nltk.download('punkt') 
nltk.download('stopwords')
nltk.download('wordnet')

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load SpaCy's English model for Named Entity Recognition
nlp = spacy.load("en_core_web_sm")

# Initialize the stemmer for stemming
lemmatizer = WordNetLemmatizer()

# Optional: A synonym mapping for common product-related terms (adjust as needed)
synonyms = {
    "jkt": "jacket",
    "coats": "coat",
    "puffer": "puffer jacket",
    "parka": "parka jacket",
    "sweater": "jumper",
    "outerwear": "coat"
}

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

def get_query_embedding(request: QueryRequest):
    query = request.query  # Access the query from the request
    query_embedding = model_embedding.encode(query, convert_to_tensor=True)
    return query_embedding


def search_similar_products(user_embedding, k=30):
    query_embedding = user_embedding.tolist()
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
    return list(results)  # Convert the cursor to a list


# Function to preprocess query by removing stop words and applying lemmatization
def preprocess_query(query: str) -> str:
    """
    Preprocess the input query by normalizing it, removing stop words, and extracting key terms.
    
    Args:
        query (str): The raw user query.
        
    Returns:
        str: The preprocessed query ready for reranking.
    """
    # Step 1: Convert to lowercase
    query = query.lower()

    # Step 2: Apply synonym replacement (expand product-related terms)
    query = " ".join([synonyms.get(word, word) for word in query.split()])

    # Step 3: Apply Named Entity Recognition to extract relevant entities (e.g., jackets, coats, etc.)
    doc = nlp(query)
    entities = [ent.text for ent in doc.ents]  # Extract entities
    entities_str = " ".join(entities)  # Combine into a string

    # Step 4: Tokenize the query and remove stopwords selectively
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(query)

    # We will remove stopwords, but keep words like 'want' or 'maybe'
    important_stopwords = {"want", "maybe", "new"}  # Adjust as needed
    tokens = [word for word in tokens if word not in stop_words or word in important_stopwords]

    # Step 5: Apply lemmatization to reduce words to their base form (e.g., "running" -> "run")
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Combine the tokens back into a query string
    query = " ".join(tokens)

    # Combine entities with filtered query terms
    query = entities_str + " " + query
    return query.strip()

def rerank_products(query, similar_products):
    query = preprocess_query(query)
    print(query)
    product_details = [
        f'{product.get("name", "Unknown Product")} (Color: {product.get("color", "Unknown Color")}, Price: ${product.get("price", "Unknown Price")})'
        for product in similar_products
    ]
    queries = [query]*len(product_details)
    features = tokenizer_llm(queries, product_details, padding=True, truncation=True, return_tensors="pt").to(device)

    with torch.no_grad():
        # Pass the features to the model
        outputs = model_llm(**features)
        logits = outputs.logits

        # Check the shape of the logits
        print(f"Logits shape: {logits.shape}")

        # Assuming logits shape is [batch_size, num_labels]
        if len(logits.shape) == 2:  # Correct for batch_size x num_labels
            # Get the sum of logits for each batch (or use another method as needed)
            scores = torch.sum(logits, dim=1).float()
        else:
            # Handle unexpected shapes, or raise an error
            raise ValueError(f"Unexpected logits shape: {logits.shape}")

        # Attach scores to the products
        for product, score in zip(similar_products, scores):
            product["score"] = score.item()

        # Sort the products by score
        reranked_products = sorted(similar_products, key=lambda x: x["score"], reverse=True)

    return reranked_products