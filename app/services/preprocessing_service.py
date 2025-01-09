from sentence_transformers import SentenceTransformer
import pandas as pd
from pymongo import MongoClient
import ast
import numpy as np
import os
from dotenv import load_dotenv
from io import StringIO
from fastapi import UploadFile
import random

# loading variables from .env file
load_dotenv()

model = SentenceTransformer(os.getenv("MY_EMBEDDING_MODEL"))
client = MongoClient(os.getenv("MY_URI_MONGODB"))
db = client["mydatabase"]
item_collection = db["items"]
product_collection = db["products"]

def generate_price(df, min_price=10, max_price=300):
    # Create a dictionary to store the prices for each product_code
    product_code_prices = {}

    def get_price_for_product_code(product_code):
        if product_code not in product_code_prices:
            # Generate a random price within the specified range
            price = round(random.uniform(min_price, max_price), 2)
            product_code_prices[product_code] = price
        return product_code_prices[product_code]

    # Apply the price generation function to each product based on product_code
    df['price'] = df['product_code'].apply(get_price_for_product_code)

    return df

async def read_and_parse_file(file: UploadFile) -> pd.DataFrame:
    file_content = await file.read()
    decoded_content = file_content.decode('utf-8')
    return pd.read_csv(StringIO(decoded_content))

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    categorical_columns = ['detail_desc', 'prod_name', 'product_type_name']
    df[categorical_columns] = df[categorical_columns].fillna('Unknown')
    df = df.dropna(subset=['detail_desc']).drop_duplicates(subset='article_id')
    return df

def filter_unwanted_data(df: pd.DataFrame) -> pd.DataFrame:
    unwanted_product_types = ['Bra', 'Underwear Tights', 'Socks', 'Leggings/Tights', 'Unknown']
    df = df[~df['product_type_name'].isin(unwanted_product_types)]
    df = generate_price(df)
    return df.rename(columns={
        'product_code': 'product_code',
        'prod_name': 'name',
        'product_type_name': 'category',
        'colour_group_name': 'color',
        'article_id': 'product_id',
        'detail_desc': 'description',
        'index_name': 'index_name',
        'price': 'price',
    })


def enrich_with_default_values(df: pd.DataFrame) -> pd.DataFrame:
    """Adds default values for stock, review, rating data, and image_url."""
    df['total_stock'] = 100
    df['sold_count'] = 0
    df['review_count'] = 0
    df['rating_count'] = 0
    df['avg_rating'] = 0.0
    # Add a default 'image_url' column if it does not exist
    if 'image_url' not in df.columns:
        df['image_url'] = ''
    return df

def clean_sizes(size_column):
    """Clean and parse the size column."""
    if isinstance(size_column, str):
        return [size.replace(" - Out of stock", "") for size in size_column.split(',')]
    return size_column
    

async def clean_file(file: UploadFile):
    try:
        df = await read_and_parse_file(file)
        df = preprocess_dataframe(df)
        df = filter_unwanted_data(df)
        df = enrich_with_default_values(df)
        return df
    except Exception as e:
        raise ValueError(f"Failed to clean the file: {str(e)}")


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

