import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", api_key=GEMINI_API_KEY)

prompt_template ="""
Question: 11 factories can make 22 cars per hour. How much time would it take 22 factories to make 88 cars?
answer Let me solve this step by step:
1. A factory can make 22/11=2 cars per hour.
2. 22 factories can make 22*2=44 cars per hour.
3. Making 88 cars would take 88/44=2 hours.

Therefore, the anwswe is 2 hours.

Now, slove this new problem using the same step-by-step approach:
Question: {question}
Answer:
"""
