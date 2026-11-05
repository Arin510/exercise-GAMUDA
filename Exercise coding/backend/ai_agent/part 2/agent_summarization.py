import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.messages import HumanMessage

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", api_key=GEMINI_API_KEY)

agent = create_agent(
    model = llm,
    checkpointer= InMemorySaver (),
    middleware=[
        SummarizationMiddleware(
            model = llm,
            trigger = ("tokens", 100),
            keep=("messages",1)
        )
    ],
    system_prompt="You are a helpful assistant."
)

config = {"configurable": {"thread_id": "1"}}
response = None
print("\nStarting chat... Type 'quit' or 'exit' to end the conversation")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["quit","exit"]:
        break

    if not user_input.strip():
        print("Please enter a message.")
        continue
    try:
        response = agent.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config
        )

        final_message = response ["messages"][-1]
        print(f"AI: {final_message.text}")

    except Exception as e:
        print(f"An error occured: {e}")

print("\nChat ended.")

if response:
    print("\nChat ended.")
    print(response["messages"][0].content)