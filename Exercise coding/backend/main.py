# backend/main.py
import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

app = FastAPI()

# Enable CORS so your frontend HTML can communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize LangChain model & prompt template
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=GEMINI_API_KEY)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that translates {input_language} to {output_language}. Provide the direct translation only."),
    ("human", "{text_input}")
])

chain = prompt | llm | StrOutputParser()

# Define Pydantic request body model
class TranslationRequest(BaseModel):
    input_language: str
    output_language: str
    text_input: str

@app.post("/translate")
async def translate_text(request: TranslationRequest):
    try:
        translated = chain.invoke({
            "input_language": request.input_language,
            "output_language": request.output_language,
            "text_input": request.text_input
        })
        return {"translated_text": translated.strip()}
    except Exception as e:
        return {"error": str(e)}