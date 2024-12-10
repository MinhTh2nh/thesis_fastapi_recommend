import os
from pymongo import MongoClient
from dotenv import load_dotenv
from bson import ObjectId

# loading variables from .env file
load_dotenv()

client = MongoClient(os.getenv("MY_URI_MONGODB"))
db = client["mydatabase"]
users_collection = db["users"]
orders_collection = db["orders"]
cart_collection = db["carts"]

def get_user_data(user_id):
    user_id_obj = ObjectId(user_id)
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

    return user, orders, cart_items, search_history
