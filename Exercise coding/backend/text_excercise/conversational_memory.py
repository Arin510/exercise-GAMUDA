import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

# Load API key from .env file
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Set up the model (Exercise 1: Try changing temperature between 0.0 and 2.0)
llm = ChatGoogleGenerativeAI(
    model="gemma-4-26b-a4b-it", # or "gemini-1.5-flash"based on the google api
    temperature=0.0,
    api_key=GEMINI_API_KEY
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer all questions to the best of your ability."),  # <--- ADD COMMA HERE!
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])


chain = prompt | llm | StrOutputParser()
history = []
print("\nStarting chat... Type 'quit' or 'exit' to end the conversation.")
while True:
    user_input = input ("You:")
    if user_input.lower() in ["quit","exit"]:
        break
    if not user_input.strip():
        print("Please enter a message.")
        continue
    try:
        response = chain.invoke({"input": user_input, "history": history})
        # Save chat history to remember context
        history.append(HumanMessage(content=user_input))
        history.append(AIMessage(content=response))
        print(f"AI: {response}")
    except Exception as e: 
        print(f"An error occurred: {e}")

print("\nChat ended.")



