from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_chunk(pdf_path: str):
    # Load the PDF
    pdf_document = PyPDFLoader(pdf_path).load()

    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(pdf_document)

    return chunks