import os
import streamlit as st
import pickle
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env (especially openai api key)

st.title("AskVarsity: Varsity Q&A Tool 📈")

main_placeholder = st.empty()
llm = OpenAI(temperature=0.9, max_tokens=1000)

file_path = "faiss_store_openai.pkl"

pdf_folder_path = "resources/"

# Load multiple files
loaders = [UnstructuredPDFLoader(os.path.join(pdf_folder_path, fn)) for fn in os.listdir(pdf_folder_path)]
main_placeholder.text("Data Loading...Started...✅✅✅")

all_documents = []

for loader in loaders:
    print("Loading raw pdf..." + loader.file_path)
    raw_documents = loader.load()

    main_placeholder.text("Text Splitter...Started...✅✅✅")

    ## Splitting the pdf text
    text_splitter = RecursiveCharacterTextSplitter(
    separators = ["\n\n","\n"," "],
    chunk_size=1000,
    chunk_overlap=200
    )
    documents = text_splitter.split_documents(raw_documents)
    all_documents.extend(documents)

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(all_documents, embeddings)
main_placeholder.text("Embedding Vector Started Building...✅✅✅")
time.sleep(2)

# Save the FAISS index to a pickle file
with open(file_path, "wb") as f:
    pickle.dump(vectorstore, f)

query = main_placeholder.text_input("Question: ")

if os.path.exists(file_path):
    with open(file_path, "rb") as f:
        vectorIndex = pickle.load(f)

        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorIndex.as_retriever())

        result = chain({"question": query}, return_only_outputs=True)
        # result will be a dictionary of this format --> {"answer": "", "sources": [] }
        st.header("Answer")
        st.write(result["answer"])

        # Display sources, if available
        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            sources_list = sources.split("\n")  # Split the sources by newline
            for source in sources_list:
                st.write(source)
            st.subheader("Source Website - :blue[https://zerodha.com/varsity/]")