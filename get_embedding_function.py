from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv

def get_embedding_function():
    embeddings = OpenAIEmbeddings(
          api_key=os.getenv("OPENAI_API_KEY")
    )
    return embeddings
