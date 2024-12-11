# loading variables from .env file
def get_user_embeddings_context(processed_user_data):
    all_products_descriptions = []  # List to store descriptive text

    # Extract product descriptions from orders
    order_products = processed_user_data.get("order_products", [])
    for order in order_products:
        product_id = order.get("productId")
        quantity = order.get("quantity", 1)
        size = order.get("size", "N/A")
        price = order.get("price", "unknown")
        if product_id:
            all_products_descriptions.append(
                f"User purchased product {product_id} (size {size}, quantity {quantity}, price {price})."
            )

    # Extract product descriptions from cart items
    cart_products = processed_user_data.get("cart_products", [])
    for cart_item in cart_products:
        product_id = cart_item.get("productId")
        if product_id:
            all_products_descriptions.append(f"User currently has product {product_id} in their cart.")

    # Extract search history as descriptive text
    search_history = processed_user_data.get("search_history", [])
    for search_item in search_history:
        all_products_descriptions.append(f"User searched for {search_item}.")

    # Combine all descriptive texts into a single context
    if not all_products_descriptions:
        return None  # Handle case where no data is available

    session_context = " ".join(all_products_descriptions)  # Combine descriptions into a single string
    # Generate embeddings for the combined context
    return session_context