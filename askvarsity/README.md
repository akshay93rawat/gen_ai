
# AskVarsity: Varsity Q&A Tool 📈 

Discover instant enlightenment with #AskVarsity! Effortlessly retrieve information by directly questioning our trained model, skipping the need to navigate through lengthy Varsity tutorials.(https://zerodha.com/varsity/)
![](rockybot.jpg)

## Features

- Load multiple PDFs from a location to fetch the content.
- Process the pdf content through LangChain's UnstructuredPDF Loader
- Construct an embedding vector using OpenAI's embeddings and leverage FAISS, a powerful similarity search library, to enable swift and effective retrieval of relevant information
- Interact with the LLM's (Chatgpt) by inputting queries and receiving answers along with source PDFs info.


## Installation

1.Clone this repository to your local machine using:

```bash
  git clone https://github.com/akshay93rawat/gen_ai.git
```
2.Navigate to the project directory:

```bash
  cd askvarsity
```
3. Install the required dependencies using pip command.

4.Set up your OpenAI API key by creating a .env file in the project root and adding your API

```bash
  OPENAI_API_KEY=your_api_key_here
```
## Usage/Examples

1. Run the Streamlit app by executing:
```bash
streamlit run main.py

```

2.The web app will open in your browser.

- At the centre, under the question message you can ask questions directly.

- Observe the system as it performs text splitting, generates embedding vectors, and efficiently indexes them using FAISS.

- The embeddings will be stored and indexed using FAISS, enhancing retrieval speed.

- The FAISS index will be saved in a local file path in pickle format for future use.

- One can also expand the coverage by adding more pdfs 

## Project Structure

- main.py: The main Streamlit application script.
- requirements.txt: A list of required Python packages for the project.
- faiss_store_openai.pkl: A pickle file to store the FAISS index.
- .env: Configuration file for storing your OpenAI API key.
- data_process.ipnb: Sample end-to-end data process