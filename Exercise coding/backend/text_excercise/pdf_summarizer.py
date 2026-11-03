import os
from io import BytesIO
from pypdf import PdfReader
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

app = FastAPI()
load_dotenv()

# Initialize Gemini 3.5 Flash-Lite model
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    api_key=os.getenv("GEMINI_API_KEY")
)

# Prompt template for text summarization
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that summarizes documents."),
    ("human", "Write a concise summary of the following:\n\n{text}\n\nCONCISE SUMMARY:")
])

chain = prompt_template | llm | StrOutputParser()

def read_pdf_file(file_contents: BytesIO) -> str:
    try:
        pdf_reader = PdfReader(file_contents)
        return "\n".join(page.extract_text() or "" for page in pdf_reader.pages)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading PDF: {e}")

@app.post("/summarize")
async def summarize_pdf(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    contents = await file.read()
    text = read_pdf_file(BytesIO(contents))
    
    if not text.strip():
        raise HTTPException(status_code=400, detail="No readable text found.")
    
    try:
        summary = await chain.ainvoke({"text": text})
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="An error occurred while summarizing the document.")
    
    return {"summary": summary}