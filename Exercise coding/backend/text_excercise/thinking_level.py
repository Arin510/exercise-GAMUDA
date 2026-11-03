import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
THINKING_LEVEL = "low"

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite", # or "gemini-1.5-flash"based on the google api
    temperature=0.0,
    api_key=os.getenv("GEMINI_API_KEY"),
    thinking_level = THINKING_LEVEL

)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{question}")
])

chain = prompt | llm | StrOutputParser()

response = chain.invoke({
    "question": "Provide a list of 3 famous pysicists and their key contributions"
})

print(f"Thinking level :{THINKING_LEVEL}")
print(f"Response\n{response}")