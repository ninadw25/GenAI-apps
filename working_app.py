import streamlit as st
import cassio
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper

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

# File Upload for PDF
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    try:
        # Save uploaded file temporarily
        file_path = f"temp_{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load and process PDF
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        # Text splitting
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=500, 
            chunk_overlap=0
        )
        doc_splits = text_splitter.split_documents(docs)

        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # Initialize AstraDB vector store
        astra_vector_store = Cassandra(
            embedding=embeddings,
            table_name="qa_mini_demo",
            session=None,
            keyspace=None
        )

        # Add documents to vector store
        astra_vector_store.add_documents(doc_splits)
        st.write(f"Inserted {len(doc_splits)} documents.")

        # Create vector store index and retriever
        astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)
        retriever = astra_vector_store.as_retriever()

        # Query interface
        query = st.text_input("Enter your query:")
        if st.button("Search"):
            try:
                results = retriever.invoke(query, ConsistencyLevel="LOCAL_ONE")
                st.write("Search Results:")
                st.write(results)
            except Exception as e:
                st.error(f"Search error: {e}")

        # Cleanup
        import os
        os.remove(file_path)

    except Exception as e:
        st.error(f"Error processing file: {e}")