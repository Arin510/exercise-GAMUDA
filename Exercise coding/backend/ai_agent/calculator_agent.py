import os
import requests
from dotenv import load_dotenv
from tavily import TavilyClient
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from typing import Dict, Any  

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", api_key = GEMINI_API_KEY)

#eval(expression) never use it on real apps where the strangers type input
#its dangerous if using to real app easer the attacker to colect the file and open env or API
@tool("calculator", description = "Performs mathematical calculations.")
def tool1 (expression: str) -> str:
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"

@tool("web_search", description = "Search the web for information.")
def tool2 (query : str) -> Dict[str , Any]|str :
    try:
        return tavily_client.search(query)
    except Exception as e:
        return f"Error: {e}"

@tool("currentcy_exchanging", description = "Get the latest currency rate. Use currency code like USD, MYR, EUR, SGD. ")
def tool3 (base : str, target : str) -> Dict[str , Any]|str :
    try:
        url = f"https://api.frankfurter.dev/v1/latest?base={base}&symbols={target}"
        response = requests.get(url)
        return response.json()
    except Exception as e:
        return f"Error: {e}"
    
tools = [ tool1, tool2, tool3]
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a helpfil assistant."
)

question = HumanMessage(content="What is the weather in Kuala Lumpur today?")
question = HumanMessage(content="What is the current exchange rate from US dollars to Malaysian Ringgit?")
response = agent.invoke ({"messages":[question]})

final_message = response ['messages'][-1]
print(final_message.text)

print("\n--- All steps ---")
for msg in response['messages']:
    msg.pretty_print()