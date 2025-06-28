import streamlit as st
import os
import ollama
#ollama.base_url = "http://host.docker.internal:11434"
#ollama_client = ollama.Client(host=ollama.base_url)

import openai
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

curr_dir = os.getcwd()

# Check if the chroma database file exists
chroma_db_file = os.path.join(curr_dir, "chroma.sqlite3")

# Define function to initialize or load the VectorStore
def initialize_vectorstore_from_web(link):
    if os.path.isfile(chroma_db_file):
        # Loading Vectorstore if present.
        embeddings = OllamaEmbeddings(model="mistral")
        vectorstore = Chroma(persist_directory=curr_dir, embedding_function=embeddings)
        st.write("Vectorstore loaded.")
        return vectorstore 
    else:
        loader = WebBaseLoader(web_paths=(link,))
        docs = loader.load()
        return create_vectorstore(docs)

## Splitting text so that the model can consume it like humans consume food in chunks.
def create_vectorstore(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
    splits = text_splitter.split_documents(docs)

     # Using OllamaEmbeddings and those formatted docs to create and store Embeddings in
    # Chromadb vector database.
    embeddings = OllamaEmbeddings(model="mistral")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=curr_dir)
    vectorstore.persist()
    st.success("✅ Vectorstore created and saved.")
    return vectorstore

def load_existing_vectorstore():
    embeddings = OllamaEmbeddings(model="mistral")
    return Chroma(persist_directory=curr_dir, embedding_function=embeddings)

# Define function to format documents to feed into the model.
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def generate_context(history):
    return "\n".join(history)

def ollama_generate_question(question, history):
    prompt = f"""Rephrase the question based on this chat history. Do not mention the history. Just return the question in double quotes.
Chat history: {history}
Question: {question}
"""
    response = ollama.chat(model='mistral', messages=[{'role': 'user', 'content': prompt}])
    return response['message']['content']

# Define function to generate a response from Ollama LLM.

def ollama_llm(question, context, history):
    prompt = f"Question: {question}\n\nContext: {context}\n\nChat History: {history}"

    response = ollama.chat(model="mistral", messages=[{'role': 'user', 'content': prompt}])
    return response['message']['content']

def openai_generate_question(question, history):
    prompt = f"""Rephrase the question based on this chat history. Do not mention the history. Just return the question in double quotes.
Chat history: {history}
Question: {question}
"""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response['choices'][0]['message']['content']

def openai_llm(question, context, history):
    prompt = f"Question: {question}\n\nContext: {context}\n\nChat History: {history}"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response['choices'][0]['message']['content']

# # Define Rag Chain
def rag_chain(question, vectorstore, history, llm_choice):
    st.session_state.conversation_history.append(question) # Add the current question to the conversation history
    history_text = generate_context(st.session_state.conversation_history)

    if llm_choice == "Ollama Mistral":
        question = ollama_generate_question(question, history_text)# retrieving better question for convo-awareness

        retriever = vectorstore.as_retriever()
        docs = retriever.invoke(question)
        context = format_docs(docs)
        return ollama_llm(question, context, history_text)

    elif llm_choice == "OpenAI GPT":
        question = openai_generate_question(question, history_text)
        retriever = vectorstore.as_retriever()
        docs = retriever.invoke(question)
        context = format_docs(docs)
        return openai_llm(question, context, history_text)

# Streamlit UI 


st.set_page_config(page_title="RAG ChatBot", layout="wide")
st.title("RAG ChatBot")


llm_choice = st.selectbox("Choose Model", ["Ollama Mistral", "OpenAI GPT"])

if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Web URL Input
link = st.text_input("Enter webpage URL:")
if st.button("Load URL"):
    if link.strip().startswith("http"):
        st.session_state.vectorstore = initialize_vectorstore_from_web(link)
    else:
        st.error("\u274c Please enter a valid URL (http/https).")

# Chat history display
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

user_input = st.chat_input("You:")
if user_input:
    if user_input.strip().lower() == "bye":
        st.chat_message("assistant").markdown("👋 Bye! See you again soon.")
        st.session_state.messages.append({"role": "assistant", "content": "👋 Bye! See you again soon."})
        st.stop()

    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    if st.session_state.vectorstore is not None:
        reply = rag_chain(user_input, st.session_state.vectorstore, st.session_state.conversation_history, llm_choice)
        st.chat_message("assistant").markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
    else:
        st.error("\u26a0\ufe0f Please load a valid URL first.")
