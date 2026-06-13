import loader
import embedder
import retriever
from dotenv import load_dotenv
from generator import generate_answer
import gradio as gr

load_dotenv()

vector_store = None
def ingest_pdf(pdf_file):
    pdf_path = pdf_file.name
    # Load the PDF and split it into chunks
    chunks =loader.load_and_chunk(pdf_path)

    global vector_store

    # Create vector store from chunks
    vector_store = embedder.create_vector_store(chunks)

    return "PDF ingested and vector store created."


def answer_question(question: str):
    if vector_store is None:
        return "Please ingest a PDF first."

    # Retrieve relevant chunks from the vector store
    retrieved_chunks = retriever.retrieve(vector_store, question)

    # Generate answer using the retrieved chunks as context
    answer = generate_answer(question, retrieved_chunks)

    return answer

# Gradio UI — this part you can lean on AI for
with gr.Blocks() as app:
    gr.Markdown("# RAG from Scratch")
    with gr.Row():
        pdf_input = gr.File(label="Upload PDF")
        ingest_btn = gr.Button("Ingest")
    status = gr.Textbox(label="Status")
    question_input = gr.Textbox(label="Ask a question")
    ask_btn = gr.Button("Ask")
    answer_output = gr.Textbox(label="Answer")

    ingest_btn.click(ingest_pdf, inputs=pdf_input, outputs=status)
    ask_btn.click(answer_question, inputs=question_input, outputs=answer_output)

if __name__ == "__main__":
    app.launch()
