import streamlit as st
import cassio
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain.embeddings import CohereEmbeddings

# Streamlit UI
st.title("LangGraph with AstraDB - Streamlit App")

# User inputs for AstraDB Connection
ASTRA_DB_APPLICATION_TOKEN = st.text_input("Enter AstraDB Token:", type="password")
ASTRA_DB_ID = st.text_input("Enter AstraDB ID:")

if st.button("Connect to AstraDB"):
    try:
        cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)
        st.success("Connected to AstraDB successfully!")
    except Exception as e:
        st.error(f"Error: {e}")

# User input for URLs
urls = st.text_area("Enter URLs (comma-separated):").split(",")

if st.button("Load and Process URLs"):
    loader = WebBaseLoader(urls)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    
    # Initialize Chroma
    vectorstore = Chroma.from_documents(docs, CohereEmbeddings())
    st.success("Documents processed and indexed successfully!")

# Query Input
query = st.text_input("Enter your query:")
if st.button("Search"):
    if vectorstore:
        results = vectorstore.similarity_search(query)
        for res in results:
            st.write(res.page_content)
    else:
        st.error("Vectorstore not initialized.")
