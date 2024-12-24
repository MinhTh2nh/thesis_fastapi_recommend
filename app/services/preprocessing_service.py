from sentence_transformers import SentenceTransformer
import pandas as pd
from pymongo import MongoClient
import ast
import numpy as np
import os
from dotenv import load_dotenv

# loading variables from .env file
load_dotenv()

model = SentenceTransformer(os.getenv("MY_EMBEDDING_MODEL"))
client = MongoClient(os.getenv("MY_URI_MONGODB"))
db = client["mydatabase"]
item_collection = db["items"]
product_collection = db["products"]

def clean_sizes(size_column):
    """Clean and parse the size column."""
    if isinstance(size_column, str):
        return [size.replace(" - Out of stock", "") for size in size_column.split(',')]
    return size_column
    
def parse_description(description_column):
    """Parse the description column if it is a string representation of a list."""
    try:
        if isinstance(description_column, str):
            return ast.literal_eval(description_column)
        return description_column
    except:
        return []   # Return an empty list if parsing fails
    
def clean_file(files):
    """Clean and preprocess uploaded files."""
    try:
        # Read content from the first uploaded file
        file_content = files[0].file.read().decode('utf-8')
        df = pd.read_csv(pd.compat.StringIO(file_content))  # Read CSV content into a DataFrame

        # Clean and preprocess data
        df['size'] = df['size'].apply(clean_sizes)
        df.rename(columns={'sku': 'product_id'}, inplace=True)
        df = df.dropna()
        df = df.drop_duplicates(subset='product_id', keep='first')
        df = df.reset_index(drop=True)
        df['description'] = df['description'].apply(parse_description)

        return df
    except Exception as e:
        raise ValueError(f"Failed to clean the file: {str(e)}")


def generate_chunking_products(df):
    """Generate products from the cleaned file and insert them into the database."""
    products = []

    for _, row in df.iterrows():
        # Handle images column
        if isinstance(row['images'], str):
            try:
                images_list = ast.literal_eval(row['images'])
            except (ValueError, SyntaxError):
                images_list = []  # Default to an empty list if parsing fails
        else:
            images_list = []

        # Prepare sizes and calculate total stock
        sizes = [{"size_name": size, "stock": 10} for size in row['size']]
        total_stock = sum(size["stock"] for size in sizes)

        # Create product document
        product_document = {
            "name": row['name'],
            "sizes": sizes,
            "category": row['category'],
            "price": row['price'],
            "color": row['color'],
            "product_id": row['product_id'],
            "description": row['description'],
            "images": images_list,
            "total_stock": total_stock,
            "sold_count": 0,
            "review_count": 0,
            "rating_count": 0,
            "avg_rating": 0.0,
        }

        # Insert document into MongoDB
        product_result = product_collection.insert_one(product_document)
        products.append(product_document)

        print(f"Inserted product with product_id: {row['product_id']} and total stock of {total_stock}.")

    return products

def generate_embeddings(products):
    embedded_products = []
    for product in products:
        try:
            description_text = product.get('description', '')
            color_text = product.get('color', '')
            price_text = f"Price: {product.get('price', 'Unknown')}"
            full_description = f"{description_text} {color_text} {price_text}"
            embedding = model.encode(full_description).tolist()
            existing_product = product_collection.find_one({"_id": product["_id"]})
            existing_embedding = existing_product.get("description_embedding")
            if existing_embedding:
                if existing_embedding == embedding:
                    print(f"Product ID {product['_id']}: Embedding is unchanged. Skipping update.")
                    continue  
            else:
                print(f"Product ID {product['_id']}: No embedding found. Updating now.")
            product_collection.update_one(
                {"_id": product["_id"]},
                {"$set": {"description_embedding": embedding}}
            )
            response_product = {
                "_id": str(product["_id"]),
                "name": product.get("name", ""),
                "product_id": product.get("product_id", 0),
                "category": product.get("category", ""),
                "price": product.get("price", ""),
                "color": product.get("color", ""),
                "sizes": product.get("sizes", []),
                "description": [item[list(item.keys())[0]] for item in product.get("description", [])],
                "images": product.get("images", []),
                "total_stock": product.get("total_stock", 0),
                "sold_count": product.get("sold_count", 0),
                "review_count": product.get("review_count", 0),
                "rating_count": product.get("rating_count", 0),
                "avg_rating": product.get("avg_rating", 0.0),
                "description_embedding": embedding,
                "reviews": product.get("reviews", []),
                "total_rating": product.get("total_rating", 0),
            }
            embedded_products.append(response_product)
        except Exception as e:
            print(f"Failed to process product ID {product.get('_id', 'Unknown')}: {e}")
    return embedded_products

