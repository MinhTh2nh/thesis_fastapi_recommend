import os
import string
import re
from pymongo import MongoClient
from dotenv import load_dotenv
from bson.objectid import ObjectId
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize the stemmer and lemmatizer
lemmatizer = WordNetLemmatizer()

# loading variables from .env file
load_dotenv()

client = MongoClient(os.getenv("MY_URI_MONGODB"))
db = client["mydatabase"]
products_collection = db["products"]
users_collection = db["users"]
orders_collection = db["orders"]
cart_collection = db["carts"]

# Helper function to retrieve product name
def get_product_name(product_id):
    product = products_collection.find_one({"_id": product_id})
    if product:
        return product.get("name", "Unknown Product")
    return "Unknown Product"

# Helper function to clean text
def clean_text(text):
    """
    Clean input text by removing punctuation, numbers, and stopwords.

    Args:
        text (str): Raw input text.

    Returns:
        cleaned_text (str): Processed and cleaned text.
    """
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation and numbers
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r'\d+', '', text)
    # Tokenize text
    text_tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in text_tokens if word not in stop_words]
    # Apply lemmatization if specified
    filtered_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    # Join the processed tokens into a single string
    cleaned_text = " ".join(filtered_tokens)
    return cleaned_text

# Remove duplicates by converting the list to a set
def remove_duplicates(text):
    tokens = text.split()  # Split text into words
    unique_tokens = set(tokens)  # Remove duplicates by converting to a set
    cleaned_text = " ".join(unique_tokens)  # Join the unique tokens back into a string
    return cleaned_text

# Function to process user data
def preprocess_user_data(orders, cart_items, search_history):
    """
    Preprocess user data including search history, order history, and cart data.
    The order history takes precedence, followed by cart, and then search history.
    Args:
        orders (list): List of order documents.
        cart_items (list): List of cart documents.
        search_history (list): List of search history strings.
    Returns:
        combined_data (str): Cleaned and combined data from orders, cart, and search history.
    """
    processed_data = []
    # Process search history: Join all search terms and clean up (lowest priority)
    if search_history:
        processed_search_history = " ".join(search_history)
        processed_search_history = clean_text(processed_search_history)
        processed_data.append(processed_search_history)  # Lowest priority
    # Process cart products: Join product names and clean up (middle priority)
    if cart_items:
        cart_descriptions = []
        for cart_item in cart_items:
            for item in cart_item['items']:
                product_name = get_product_name(item['productId'])
                cart_descriptions.append(f"{product_name}")
        processed_cart_products = " ".join(cart_descriptions)
        processed_cart_products = clean_text(processed_cart_products)
        processed_data.append(processed_cart_products)  # Middle priority
    # Process order history: Join product names and clean up (highest priority)
    if orders:
        order_descriptions = []
        for order in orders:
            for item in order['items']:
                product_name = get_product_name(item['productId'])
                order_descriptions.append(f"{product_name}")
        processed_order_history = " ".join(order_descriptions)
        processed_order_history = clean_text(processed_order_history)
        processed_data.append(processed_order_history)  # Highest priority

    # Combine all processed data
    combined_data = " ".join(processed_data)
    # Remove duplicates from the combined data
    combined_data = remove_duplicates(combined_data)
    return combined_data


# Function to fetch user data from the database
def get_user_data(user_id):
    user_id_obj = ObjectId(user_id)
    
    try:
        # Fetch user details including search history
        user = users_collection.find_one({"_id": user_id_obj})
        if not user:
            print(f"User with ID {user_id} not found.")  # Log if the user is not found
            return None, [], [], []  # Return empty lists if user not found

        # Fetch order history and cart items
        orders = list(orders_collection.find({"userId": user_id_obj})) or []
        cart_items = list(cart_collection.find({"userId": user_id_obj})) or []

        # Extract search history from the user document
        search_history = user.get("search_history", []) or []

        return orders, cart_items, search_history
    except Exception as e:
        print(f"Error fetching user data: {e}")
        return None, [], [], []

