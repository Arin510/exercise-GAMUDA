import os, base64
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    api_key=os.getenv("GEMINI_API_KEY")
)
video_file_path="farmer.mp4"
with open(video_file_path, "rb") as video_file:
    encoded_video = base64.b64encode(video_file.read()).decode()

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert video analyst. Analyze videos carefully and provide detailed descriptions."),
    ("human", [
        {
            "type": "text",
            "text": "{analysis_request}"
        },
        {
            "type": "media",
            "data": "{video_data}",
            "mime_type": "{mime_type}"
        }
    ])
])

chain = prompt | llm | StrOutputParser()
requests = [
      "Describe what's happening in this video.",   # original prompt from the slides
        "Give a detailed summary of this video.",
        "Summarize this video in a few short bullets.",
        "Generate a paragraph that summarizes this video, with corresponding timecodes.",
        "Summarize this video. Then create a quiz with an answer key based on the information in this video.",
    ]

for request in requests:
    print("Analyzing video...")
# response = chain.invoke({
#     "analysis_request": "describe whats's happening in this video.",
#     "youtube_url": youtube link --> ni kalau mau copy link youtube 
# })

response = chain.invoke({
    "analysis_request": "describe whats's happening in this video.",
    "video_data": encoded_video,
    "mime_type": "video/mp4"
})

print("Analysis completed")
print("\nResults:\n" + "-" * 40)
print(response)
