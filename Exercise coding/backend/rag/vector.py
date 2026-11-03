import os
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_mongodb import MongoDBAtlasVectorSearch
from embeddings import embeddings

load_dotenv()
MONGODB_URI = os.getenv("MONGODB_ATLAS_CLUSTER_URI")

#connect to MongoDB
client = MongoClient(MONGODB_URI)

#define database and collection names
DB_NAME= "test_db"
COLLECTION_NAME = "test_collection"
ATLAS_VECTOR_SEARCH_INDEX_NAME="test-index-1"
MONGODB_COLLECTION = client[DB_NAME][COLLECTION_NAME]

#instantiate the victor store
vector_store = MongoDBAtlasVectorSearch(
    collection= MONGODB_COLLECTION,
    embedding= embeddings,
    index_name= ATLAS_VECTOR_SEARCH_INDEX_NAME,
    relevance_score_fn="cosine",
)


# Create the vector search index
print("Creating vector search index...")
vector_store.create_vector_search_index(
    dimensions=768, wait_until_complete=60
)

print("Vector store Created!")

client.close()