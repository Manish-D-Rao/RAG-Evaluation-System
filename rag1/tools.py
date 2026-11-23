from .indexing import vector_store
from langchain.tools import tool

last_retrieved_context = []

@tool
def collection_info(query: str) -> str:
    """
    Retrieve information from the vector store and return the relevant information based on the query.
    """

    global last_retrieved_context
    docs = vector_store.similarity_search(query, k=4)

    last_retrieved_context = [doc.page_content for doc in docs]

    return "\n\n".join(last_retrieved_context)