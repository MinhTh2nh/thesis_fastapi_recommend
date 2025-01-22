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
        expected_columns = {
            'product_code': 'product_code',
            'name': 'name',
            'category': 'category',
            'color': 'color',
            'product_id': 'product_id',
            'description': 'description',
            'index_name': 'index_name',
            'price': 'price',
        }
        if set(expected_columns.values()).issubset(df.columns):
            print("The file already has the expected structure. Skipping preprocessing steps.")
            # Enrich with default values for processed data
            df = enrich_with_default_values(df)
        else:
            print("The file is raw and requires preprocessing.")
            df = preprocess_dataframe(df)
            df = filter_unwanted_data(df)
            df = enrich_with_default_values(df)
        return df
    except Exception as e:
        raise ValueError(f"Failed to clean the file: {str(e)}")

