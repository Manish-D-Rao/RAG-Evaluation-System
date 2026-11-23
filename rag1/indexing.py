import os
from dotenv import load_dotenv
from .file_reader import extract_paragraphs
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

load_dotenv()
NEXUS_API_KEY = os.getenv("NEXUS_API_KEY")
NEXUS_BASE_URL = os.getenv("NEXUS_BASE_URL")

embedding = OpenAIEmbeddings(model="text-embedding-3-small", base_url=NEXUS_BASE_URL, api_key=NEXUS_API_KEY)

vector_store = InMemoryVectorStore(embedding)

chunks = extract_paragraphs("data/dalai_lama_purpose_of_life_essay.pdf")

vector_store.add_texts(chunks)
print(f"Number of chunks added to the vector store: {len(chunks)}")