from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

def generate_answer(question: str, retrieved_chunks: list):
    # Build context string from retreived chunks
    context = "\n\n".join([chunk.page_content for chunk in retrieved_chunks])

    # Build prompt - system message + context + question
    # system_message = SystemMessage(content="""You are finacial advisor who will answer 
    #                                the question based only on the provided context.  If you donnt know say you dont know. 
    #                                Always use all the information provided in the context.""")

    system_message = SystemMessage(content="""You are a financial portfolio analyst.
When asked to categorize ETF holdings, use these definitions:
- Foundational: broad market, all-in-one, invested on sp500 index or diversified core ETFs
- Growth: tech-heavy, innovation, high-growth sector ETFs
- Value: dividend-focused, income, or value-factor ETFs
- Fixed Income: bond ETFs of any duration or type

Classify each holding accordingly and calculate allocation percentages.
                                   """)
    human_message = HumanMessage(content=f"""Context:
                                 {context}
                                 
                                 Question: {question}""")
    
    # Create chat model
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
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