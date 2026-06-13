from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

def create_vector_store(chunks):
    # Create embedding model
    embeddings = OpenAIEmbeddings()

    # Store chunks in vector store
    return Chroma.from_documents(chunks, embeddings, collection_name="pdf_chunks")

