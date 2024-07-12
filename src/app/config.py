import os
from langchain_google_genai import ChatGoogleGenerativeAI

def load_config():
    model = os.getenv("MODEL")
    llm = ChatGoogleGenerativeAI(
        temperature=0.7,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        model=model,
        max_tokens=300,
    )
    return model, llm
