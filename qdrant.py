import os
import json
import hashlib
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from openai import AzureOpenAI

# --- Qdrant Setup ---
qdrant_client = QdrantClient(
    url="your_qdrant_url",
    api_key="your_qdrant_api"
)

# --- Azure OpenAI Embeddings Setup ---
azure_endpoint = "your_azure_url"
api_key_azure = "your_azure_api"

client = AzureOpenAI(
    api_key=api_key_azure,
    azure_endpoint=azure_endpoint,
    api_version="your_api_version"
)

# --- Qdrant Collection Info ---
collection_name = "your_collection_name"
vector_dimension = 3072
distance_metric = models.Distance.COSINE

def create_or_verify_collection():
    try:
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_dimension,
                distance=distance_metric
            )
        )
        print(f"Collection '{collection_name}' created.")
    except Exception as e:
        if "already exists" in str(e):
            print(f"Collection '{collection_name}' already exists.")
        else:
            print(f"Error creating or verifying collection: {e}")

def get_embeddings(text):
    if not text or not isinstance(text, str):
        return None
    try:
        response = client.embeddings.create(
            input=[text],
            model="text-embedding-3-large"
        )
        return np.array(response.data[0].embedding).tolist()
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

def generate_vector_id(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()

# --- Load and Process JSON ---
def process_json_file(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    create_or_verify_collection()

    for i, entry in enumerate(json_data.get("data", [])):
        filename = entry.get("filename", "")
        source = entry.get("source", "")
        summary_list = entry.get("summary", [])
        summary_text = summary_list[0] if summary_list else ""

        if not summary_text:
            print(f"No summary found for index {i}. Skipping.")
            continue

        embedding = get_embeddings(summary_text)
        if embedding:
            vector_id = generate_vector_id(filename)
            metadata = {"filename": filename, "source": source}

            qdrant_client.upsert(
                collection_name=collection_name,
                points=[
                    models.PointStruct(
                        id=vector_id,
                        vector=embedding,
                        payload=metadata
                    )
                ]
            )
            print(f"Uploaded: '{filename}'")
        else:
            print(f"Failed to embed: '{filename}'")

# --- Run the script ---
if __name__ == "__main__":
    json_file_path = "raw.json"  # <- replace with actual filename
    process_json_file(json_file_path)
