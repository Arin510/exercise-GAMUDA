import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import HumanMessage

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DB_URI = os.getenv("DB_URI")
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", api_key=GEMINI_API_KEY)

with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    # checkpointer.setup()

    agent = create_agent(
        model = llm,
        checkpointer=checkpointer
    )

    config = {"configurable": { "thread_id": "1"}}

    response = agent.invoke({"messages": [HumanMessage(content="Hi! My name is Ain.")]},config)
    print(response["messages"][-1].text)