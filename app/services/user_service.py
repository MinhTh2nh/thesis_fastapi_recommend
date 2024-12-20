import os
from pymongo import MongoClient
from dotenv import load_dotenv
from bson.objectid import ObjectId

# loading variables from .env file
load_dotenv()

client = MongoClient(os.getenv("MY_URI_MONGODB"))
db = client["mydatabase"]
products_collection = db["products"]
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

    return orders, cart_items, search_history

def preprocess_user_data(orders, cart_items, search_history):
    """
    Preprocess user data by retrieving product names from the products collection.

    Args:
        orders (list): List of order documents.
        cart_items (list): List of cart documents.
        search_history (list): List of search history strings.
        products_collection: MongoDB collection object for products.

    Returns:
        dict: Processed user data containing order products, cart products, and search history.
    """
    # Helper function to retrieve product details by productId
    def get_product_name(product_id):
        product = products_collection.find_one({"_id": product_id})
        return product.get("name", "Unknown Product") if product else "Unknown Product"

    # Process order items (Product Name, Quantity)
    order_products = []
    for order in orders:
        for item in order['items']:
            product_name = get_product_name(item['productId'])  # Fetch product name
            product = {
                'product_name': product_name,
                'quantity': item['quantity'],
                'size': item['size'],
                'price': item['price']
            }
            order_products.append(product)

    # Process cart items (Product Name)
    cart_products = []
    for cart_item in cart_items:
        for item in cart_item['items']:
            product_name = get_product_name(item['productId'])  # Fetch product name
            product = {
                'product_name': product_name,
                'quantity': item['quantity'],
                'size': item['size']
            }
            cart_products.append(product)

    # Process search history (if available)
    search_history_data = search_history or []  # Assuming empty if None

    # Combine all data for recommendation purposes
    processed_data = {
        'order_products': order_products,
        'cart_products': cart_products,
        'search_history': search_history_data
    }

    return processed_data
