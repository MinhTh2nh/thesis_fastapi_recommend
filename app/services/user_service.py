import os
from pymongo import MongoClient
from dotenv import load_dotenv
from bson.objectid import ObjectId

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

def preprocess_user_data(user, orders, cart_items, search_history):
    # Extract user details
    user_name = user.get("name", "Unknown")
    user_email = user.get("email", "Not Provided")
    avatar_url = user.get("avatar", "")
    # Process order items (ProductID, Quantity)
    order_products = []
    for order in orders:
        for item in order['items']:
            product = {
                'productId': str(item['productId']),  # Convert ObjectId to string
                'quantity': item['quantity'],
                'size': item['size'],
                'price': item['price']
            }
            order_products.append(product)

    # Process cart items (ProductID)
    cart_products = []
    for cart_item in cart_items:
        for item in cart_item['items']:
            price = item.get('price', 0)
            product = {
                'productId': str(item['productId']),  # Convert ObjectId to string
                'quantity': item['quantity'],
                'size': item['size'],
                'price': price
            }
            cart_products.append(product)

    # Process search history (if available)
    search_history_data = search_history or []  # Assuming empty if None

    # Combine all data for recommendation purposes
    processed_data = {
        'user_name': user_name,
        'user_email': user_email,
        'avatar': avatar_url,
        'order_products': order_products,
        'cart_products': cart_products,
        'search_history': search_history_data
    }

    return processed_data
