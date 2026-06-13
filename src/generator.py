from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

def generate_answer(question: str, retrieved_chunks: list):
    # Build context string from retreived chunks
    context = "\n\n".join([chunk.page_content for chunk in retrieved_chunks])

    # Build prompt - system message + context + question
    system_message = SystemMessage(content="""You are finacial advisor who will answer 
                                   the question based only on the provided context.  If you donnt know say you dont know. 
                                   Always use all the information provided in the context.""")

    human_message = HumanMessage(content=f"""Context:
                                 {context}
                                 
                                 Question: {question}""")
    
    # Create chat model
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    response = llm.invoke([system_message, human_message])

    return response.content


# at the bottom of generator.py temporarily
if __name__ == "__main__":
    load_dotenv()
    # fake a chunk to test without ChromaDB
    from langchain_core.documents import Document
    fake_chunks = [
        Document(page_content="The MER of XEQT is 0.20% per year."),
        Document(page_content="XEQT is suitable for long-term investors.")
    ]
    answer = generate_answer("What is the MER?", fake_chunks)
    print(answer)