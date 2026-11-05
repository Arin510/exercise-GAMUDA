import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", api_key=GEMINI_API_KEY)

agent = create_agent(
    model=llm,
    checkpointer = InMemorySaver()
)

config = {"configurable": {"thread_id":"1"}}

response = agent.invoke ({"messages": [HumanMessage(content="Hi! my name is Ain")]}, config)
print(response["messages"][-1].text)

response = agent.invoke ({"messages": [HumanMessage(content="Whay is my name?")]}, config)
print(response["messages"][-1].text)