import streamlit as st
import os
import ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings

# Get the current working directory
curr_dir = os.getcwd()

# Check if the chroma database file exists
chroma_db_file = os.path.join(curr_dir, "chroma.sqlite3")

# Define function to initialize or load the VectorStore
def initialize_or_load_vectorstore(link):
    if os.path.isfile(chroma_db_file):
        # Loading Vectorstore if present.
        embeddings = OllamaEmbeddings(model="mistral")
        vectorstore = Chroma(persist_directory=curr_dir, embedding_function=embeddings)
        st.write("Vectorstore loaded.")
    else:
        # Creating VectorStore if not present, change web_paths depending on your personal docs.
        loader = WebBaseLoader(web_paths=(link,))
        st.write("Document Loaded!")
        docs = loader.load()

        # Splitting text so that the model can consume it like humans consume food in chunks.
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
        splits = text_splitter.split_documents(docs)

        # Using OllamaEmbeddings and those formatted docs to create and store Embeddings in
        # Chromadb vector database.
        embeddings = OllamaEmbeddings(model="mistral")
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=curr_dir)
        vectorstore.persist()
        st.write("Vectorstore created and persisted(saved).")

    return vectorstore

# Define function to format documents to feed into the model.
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Define function to generate context from conversation history.
def generate_context(history):
    return "\n".join(history)

# Define function to retrieve a better question from Ollama LLm.
def ollama_generate_question(question, history):
    formatted_prompt = f"""I have sent you chat history and question,
    rephrase that question depending on the chat history. Do not mention anything about history,
    just send the question inside double quotes.
    Chat history: {history}\n\nQuestion: {question}\n"""

    response = ollama.chat(model='mistral', messages=[{'role': 'user', 'content': formatted_prompt}])
    return response['message']['content']

# Define function to generate a response from Ollama LLM.
def ollama_llm(question, context, history):
    if len(history) > 10:
        history = history[4:]

    formatted_prompt = f"Question: {question}\n\nContext: {context}\n\nPrevious inputs from user: {history}"

    response = ollama.chat(model="mistral", messages=[{'role': 'user', 'content': formatted_prompt}])
    return response['message']['content']

# Define Rag Chain
def rag_chain(question, vectorstore, history):
    conversation_history.append(question)  # Add the current question to the conversation history
    history = generate_context(conversation_history)

    question = ollama_generate_question(question, history)  # retrieving better question for convo-awareness

    retriever = vectorstore.as_retriever()
    retrieved_docs = retriever.invoke(question)
    formatted_context = format_docs(retrieved_docs)
    return ollama_llm(question, formatted_context, history)

# Main function to run Streamlit app
def main():
    st.title("Website Chat Interface")

    # Input block for entering webpage link
    link = st.text_input("Enter the link to the article:")

    # Initialize or load the VectorStore
    vectorstore = initialize_or_load_vectorstore(link)

    # Initialize conversation history
    global conversation_history
    conversation_history = []

    # Chat interface
    while True:
        user_input = st.text_input("You:")
        if user_input == "/bye":
            st.write("Exiting chat mode...")
            break
        else:
            result = rag_chain(user_input, vectorstore, conversation_history)
            st.write("Bot:", result)

if __name__ == "__main__":
    main()

