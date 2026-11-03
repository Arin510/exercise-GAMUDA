# translator.py
import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables from the .env file
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware so your frontend HTML can communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the Gemini LLM via LangChain (using gemini-1.5-flash as shown in coursework)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=GEMINI_API_KEY)

# Create a ChatPromptTemplate with system instructions and human input placeholder
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that translates {input_language} to {output_language}. Translate the provided sentence only."),
    ("human", "{text_input}")
])

# Combine prompt, model, and output parser into a LangChain chain
chain = prompt | llm | StrOutputParser()

# Define the request body structure using Pydantic
class TranslationRequest(BaseModel):
    input_language: str
    output_language: str
    text_input: str

# Define the POST endpoint for translation
@app.post("/translate")
async def translate_text(request: TranslationRequest):
    try:
        # Invoke the LangChain chain with the data sent from the frontend
        translated = chain.invoke({
            "input_language": request.input_language,
            "output_language": request.output_language,
            "text_input": request.text_input
        })
        return {"translated_text": translated.strip()}
    except Exception as e:
        return {"error": str(e)}