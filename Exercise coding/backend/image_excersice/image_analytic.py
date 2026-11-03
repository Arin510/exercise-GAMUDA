import os, base64
from fastapi import FastAPI, File, UploadFile, HTTPException
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate  
from langchain_core.output_parsers import StrOutputParser
from typing import Dict

# Load API key from .env file
load_dotenv()
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    api_key=os.getenv("GEMINI_API_KEY")
)

app = FastAPI()

def encode_image(image_content: bytes) -> str:
    return base64.b64encode(image_content).decode()

@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)) -> Dict[str, str]:

    if not file.content_type or file.content_type not in ["image/jpeg"]:
        raise HTTPException(status_code=400, detail="invalid or missing file type. Only JPEG and PNG are allowed.")

    contents =  await file.read()

    if len(contents) > 10_000_000:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 10MB.")

    image = encode_image(contents)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "you are a nutrian expert capable of analyzing food images and providing detailed nutrianal advice."),
        ("human", [
            {
                "type": "text",
                "text": """Analyze the image and provide a comprehensive nutrional breakdown and health advice. Follow these steps:

                1. Identify every food and drink item visible in the image.
                2. Estimate the portion size of each item (in grams or common units).
                3. For each item, estimate calories, protein, carbohydrates, fat, fibre and sugar.
                4. Give the total estimated calories and macronutrients for the whole meal.
                5. Point out notable vitamins/minerals and any concerns (high sodium, sugar, fried food, allergens).
                6. Rate the overall healthiness of the meal from 1 to 10 and explain why.
                7. Suggest 2-3 practical changes to make the meal healthier.
                If the image does not contain food, say so politely and stop.
                Present the answer with clear headings and a table for the nutrition breakdown.
                """ 
            },
            {
                "image_url": {
                    "url": f"data:{file.content_type};base64,{image}",
                },
            },
        ]),
    ])

    try:
        chain = prompt | llm | StrOutputParser()
        res = await chain.ainvoke({})
    except Exception as e:
        print(str(e))
        raise HTTPException(status_code=500, detail="An error occurred while analyzing the image.")
    return {"analysis": res}
