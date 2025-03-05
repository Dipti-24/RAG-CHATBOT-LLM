# RAG-CHATBOT-LLM


## Overview of the Chatbot Architecture

The provided chatbot is designed to leverage the RAG (Retrieval-Augmented Generation) framework, vectordb (Chroma vector database), Embedding (OllamaEmbeddings), and LLM (Language Model) frameworks. The chatbot combines these components to create a conversational agent capable of retrieving information and generating responses in a coherent manner and also has beta level conversational awareness.


## Explanation of Framework Utilization

1. **RAG Framework:**
   The RAG framework is utilized to retrieve relevant documents based on user queries. The chatbot uses the RAG framework to retrieve information from a vector database created using Chroma.

2. **vectordb (Chroma):**
   The vectordb component is implemented through Chroma, a vector database. It stores document embeddings generated by OllamaEmbeddings. The database is queried using the RAG framework to obtain relevant documents related to user queries.

3. **Embedding (OllamaEmbeddings):**
   OllamaEmbeddings is employed to generate embeddings for documents. These embeddings are stored in the Chroma vector database, enabling efficient retrieval of relevant information.

4. **LLM (Large Language Model):**
   The Ollama LLM (Large Language Model by ollama - Mistral) is implemented using Ollama for rephrasing user questions based on the chat history. This enhances conversational awareness by considering the context of the ongoing conversation.








## Environment Setup and Chatbot Execution Instructions

### 1. Environment Setup:
   - Create an environment using conda.
   - Ensure Python is installed on your system.
   - Unzip the Chatbot_DVM.zip into your desired location
   - Find requirements.txt file
   - Install dependencies using cmd: `pip install -r requirements.txt`


### 2. Running the Chatbot:

 run  "streamlit run new_chatbot.py"

## Using Pre-trained Chatbots:

### Test_1 or Test_2 Folders:
1. Navigate to either the Test1 or Test2 folders.
2. Run the following command in the cmd: `streamlit run new_main_chatbot.py` to experience the pre-trained chatbots.
3. Find a `chroma.sqlite3` file in both directories, containing vectordb embeddings for specific articles.
4. Sample conversations are available in the `sample.pdf`.

## Training Your Own Chatbot:

1. Create a folder with your desired name and place `new_chatbot.py` inside.
2. Modify the `web_paths` variable within the loader to include the URL of your desired webpage. Save the changes.
3. Execute the command: `python main_chatbot.py`.
4. If the Chroma database file doesn't exist, the chatbot will generate and persist a new vectorstore using the specified web documents.

## General Chatbot Setup:

1. Clone or download the chatbot code.
2. Open a terminal in the code directory.
3. Verify the existence of the Chroma database file; if absent, the chatbot will establish a new vectorstore using web documents make sure url is present in it.
4. Run the chatbot using: `Streamlit run new_chatbot.py`.
5. Engage in a chat by entering user inputs. Type "/bye" to conclude the conversation.


Note: Adjust web_paths in the code based on your personal document sources.

## Sample Conversation:

Visit Test_1 or Test_2 to look at their sample_conversations.pdf file. It contains sample conversations.



## Contacts:
	    for any query  mail me at diptimishra2402@gmail.com
            
