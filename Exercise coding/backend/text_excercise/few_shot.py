# few_shot.py
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", api_key=GEMINI_API_KEY)

# Custom Few-Shot Examples
examples = [
    {"review": "I love the food its really good and have a good price.", "sentiment": "Positive"},
    {"review": "Im so dissapointed because it not what I expect", "sentiment": "Negative"},
    {"review": "Well, its no special.", "sentiment": "Neutral"}
]

example_prompt = PromptTemplate(
    input_variables=["review", "sentiment"],
    template="Food Review: {review}\nSentiment: {sentiment}"
)

prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Analyze the food review sentiment.\nRespond with ONLY ONE WORD: Positive, Negative, Mixed, or Neutral.\n",
    suffix="Food Review: {input}\nSentiment:",
    input_variables=["input"]
)

chain = prompt | llm | StrOutputParser()
print("Food Review Sentiment Analyzer (type 'quit' to exit)")
while True:
    review = input("\nEnter food review: ").strip()
    if not review:
        continue
    if review.lower() == 'quit':
        break
    try:
        response = chain.invoke({"input": review})
        print(response.strip())
    except Exception as e:
        print(f"Error: {e}")

print("\nProgram ended.")