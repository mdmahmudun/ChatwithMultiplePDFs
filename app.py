import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

def get_text_from_pdf(pdfs):
    text = ""
    for pdf in pdfs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=2000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    # vector_store.save_local('faiss_index')
    return vector_store

def get_conversational_chain():
    prompt_template = """
    Answer the question in details, if the answer is not in provided context just say, "Answer is not available in the context", don't provide the wrong answer \n\n
    Context: \n{context}?\n
    Question: \n{question}\n
    Answer: 
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5)
    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
    chain = load_qa_chain(model, chain_type='stuff', prompt=prompt)
    return chain

def user_input(user_question, chat_history):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # new_db = FAISS.load_local('faiss_index', embeddings=embeddings, allow_dangerous_deserialization=True)
    new_db = st.session_state.vector_store
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    
    chat_history.append({"user": user_question, "bot": response['output_text']})
    return chat_history

def display_chat(chat_history):
    for chat in chat_history:
        st.markdown(f"""
            <div style='display: flex; align-items: center; margin-bottom: 10px;'>
                <div style='background-color: #ffcccb; border-radius: 50%; padding: 10px; margin-right: 10px;'>User</div>
                <div style='background-color: #f1f1f1; padding: 10px; border-radius: 10px; flex: 1;'>{chat['user']}</div>
            </div>
            <div style='display: flex; align-items: center; margin-bottom: 10px;'>
                <div style='background-color: #add8e6; border-radius: 50%; padding: 10px; margin-right: 10px;'>Bot</div>
                <div style='background-color: #f1f1f1; padding: 10px; border-radius: 10px; flex: 1;'>{chat['bot']}</div>
            </div>
        """, unsafe_allow_html=True)

def main():
    st.set_page_config('Chat PDF')
    st.header("Chat with Multiple PDFs :books:")

    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    user_question = st.text_input('Ask a Question From the PDF Files')

    
    if user_question:
        st.session_state.chat_history = user_input(user_question=user_question, chat_history=st.session_state.chat_history)

    display_chat(st.session_state.chat_history)
    
    with st.sidebar:
        st.title('Menu')
        pdf_docs = st.file_uploader('Upload your PDF Files and Click on the Submit and Process Button', accept_multiple_files=True)
        if st.button('Submit & Process', key="process"):
            with st.spinner("Processing..."):
                raw_text = get_text_from_pdf(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                st.session_state.vector_store = get_vector_store(text_chunks=text_chunks)
                st.success('Done')

    # Move text input to the bottom
    st.markdown("""
        <style>
            .stTextInput, .stButton {
                position: fixed;
                bottom: 10px;
                width: 80%;
            }
            .stButton#send {
                width: 10%;
                left: 85%;
                bottom: 10px;
            }
            .stButton#process {
                position: relative;
                width: auto;
                left: 0;
                bottom: 0;
            }
        </style>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
