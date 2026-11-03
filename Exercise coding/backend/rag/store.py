import os
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_mongodb import MongoDBAtlasVectorSearch
from embeddings import embeddings

load_dotenv()
MONGODB_URI = os.getenv("MONGODB_ATLAS_CLUSTER_URI")

client = MongoClient(MONGODB_URI)
DB_NAME = "test_db"
COLLECTION_NAME = "test_collection"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "test-index-1"
MONGODB_COLLECTION= client [DB_NAME][COLLECTION_NAME]

vector_store = MongoDBAtlasVectorSearch(
    collection=MONGODB_COLLECTION,
    embedding=embeddings,
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
    relevance_score_fn="cosine"
)

#start clean: re_embed everything on each run 
MONGODB_COLLECTION.delete_many({})
print("Previous documents cleared")

# Load the text file as one document 
def load_tex_document(file_path: str) -> list[Document]:
    with open(file_path,"r", encoding="utf-8") as f:
        text = f.read()
    return [Document(page_content=text, metadata={"source": file_path})]

docs= load_tex_document("facts.txt")

#split into chuncks of about 200 characters 
text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size=200,
    chunk_overlap=0
)
split_docs = text_splitter.split_documents(docs)

# Add the chuncks to the vector store
print("Adding documents...")
vector_store.add_documents(split_docs)
print("Documents Added")

client.close()