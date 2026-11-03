# streamlit
import os, base64
import streamlit as st
from fastapi import FastAPI, File, UploadFile, HTTPException
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate  
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    api_key=os.getenv("GEMINI_API_KEY")
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "you are a nutrian expert capable of analyzing food images and providing detailed nutrianal advice."),
        ("human", 
            [
                {"type": "text", "text" : "{input}"},
                {
                    "type": "image_url",
                    "image_url": {
                    "url": "data:{mime_type};base64,{image}",
                    },
                },
            ],
        ),
    ]
)

chain = prompt | llm | StrOutputParser()

st.title("Image Analysis Using Gemini")

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Upload your image", type=["jpg", "png"])

with col2:
    if uploaded_file is not None:
        image_bytes = uploaded_file.read()
        st.image(image_bytes)
    else:
        st.write("Welcome to the Image Analysis App!")

question = st.text_input("Enter a question")

if question and uploaded_file is not None:
    image = base64.b64encode(image_bytes).decode()
    res = chain.invoke({"input": question, "image": image, "mime_type": uploaded_file.type})
    st.write(res)