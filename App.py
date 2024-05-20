import os
from PyPDF2 import PdfReader
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import OpenAI
from langchain_community.callbacks import get_openai_callback

os.environ["OPENAI_API_KEY"] = ""

def process_text(text):
    # Split the text into chunks using langchain
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # Convert the chunks of text into embeddings to form a knowledge base
    embeddings = OpenAIEmbeddings()
    knowledgeBase = FAISS.from_texts(chunks, embeddings)

    return knowledgeBase

def main():
    st.title("SNERC Knowledge Retrieval ▒~_~R▒")

    pdfs = st.file_uploader('Upload your documents', type='pdf', accept_multiple_files=True)

    if 'knowledgeBase' not in st.session_state:
        # Text variable will store the pdf text
        text = ""

        for pdf in pdfs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()

        if text:
            # Create the knowledge base object
            st.session_state['knowledgeBase'] = process_text(text)

    if 'knowledgeBase' in st.session_state:
        query = st.text_input('Ask a question')
        cancel_button = st.button('Cancel')

        if cancel_button:
            st.stop()

        if query:
            docs = st.session_state['knowledgeBase'].similarity_search(query)
            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type='stuff')

            with get_openai_callback() as cost:
                response = chain.run(input_documents=docs, question=query)
                print(cost)

            st.write(response)


if __name__ == "__main__":
    main()
