from sentence_transformers import SentenceTransformer
import numpy as np
import os
from dotenv import load_dotenv

# loading variables from .env file
load_dotenv()

model = SentenceTransformer(os.getenv("MY_EMBEDDING_MODEL"))

def generate_user_embedding(user_data):
    user, orders, cart_items, search_history = user_data

    # Create lists of interactions (product IDs or search terms)
    order_products = [item['productId'] for order in orders for item in order['items']]
    cart_product_ids = [item['productId'] for item in cart_items for item in item['items']]
    search_keywords = search_history  # Using the search history directly

    # Combine all interactions into one list of strings
    all_interactions = order_products + cart_product_ids + search_keywords

    # Convert the interactions into text (if they aren't already text)
    interaction_texts = [str(interaction) for interaction in all_interactions]

    # Generate embeddings for each interaction
    interaction_embeddings = model.encode(interaction_texts)

    # Aggregate the embeddings by averaging
    if len(interaction_embeddings) == 0:
        return np.zeros(model.get_sentence_embedding_dimension())  # Return zero vector if no embeddings

    user_embedding = np.mean(interaction_embeddings, axis=0)

    return user_embedding
