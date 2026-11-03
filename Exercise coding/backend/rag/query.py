import os
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mongodb import MongoDBAtlasVectorSearch
from embeddings import embeddings

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MONGODB_URI = os.getenv("MONGODB_ATLAS_CLUSTER_URI")

#Initialize the LLM
llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", api_key=GEMINI_API_KEY)

client = MongoClient(MONGODB_URI)
DB_NAME = "test_db"
COLLECTION_NAME="test_collection"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "test-index-1"
MONGODB_COLLECTION = client[DB_NAME][COLLECTION_NAME]

#vector store and retriver
# k=3 returns the top 3 most relevant facts
vector_store = MongoDBAtlasVectorSearch(
    collection=MONGODB_COLLECTION,
    embedding= embeddings,
    index_name= ATLAS_VECTOR_SEARCH_INDEX_NAME,
    relevance_score_fn="cosine",
)
retriever=vector_store.as_retriever(search_kwargs={"k": 3})

#Define the system prompt with instructions for the AI
system_prompt = (
     """
    You are a friendly assistant that answers questions about fun facts.
    Answer ONLY using the information in the CONTEXT below.
    If the answer is not in the context, say: "Sorry, I don't have that information."
    Do not make up facts.
    Keep your answer short and clear.

    CONTEXT:\n{context}
    """
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

#Merge the retru=ived chuncks into one string
def format_docs(docs):
    return"\n\n". join(doc.page_content for doc in docs)

#prepare the inputs for the prompt
setup_and_retrieval = {
    "context": retriever|format_docs,
    "input": RunnablePassthrough(),
}

#Retrive, build the prompt, call the LLM, parse the text
rag_chain = setup_and_retrieval| prompt|llm|StrOutputParser()

#execute the query
print("Querying vector DB...")
response=rag_chain.invoke("Three facts about animals")
print(f"\nAI Response: {response}")

client.close()