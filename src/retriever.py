def retrieve(vector_store, question: str, k: int = 5):
    # Retrieve relevant chunks from the vector store
    return vector_store.similarity_search(question, k=k)