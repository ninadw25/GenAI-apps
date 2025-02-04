import streamlit as st
from dotenv import load_dotenv
import os
from utils.database import init_astradb_connection
from utils.document_processor import process_pdf
from services.vector_store import create_vector_store
from services.llm_service import init_llm, create_router
from services.search_service import init_wiki_search
from graph.workflow import create_workflow

def initialize_services():
    """Initialize all required services."""
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("GROQ_API_KEY not found in .env file")
        return None
    
    llm = init_llm(groq_api_key)
    question_router = create_router(llm)
    # Add debug print
    wiki_tool = init_wiki_search()
    print("Wiki tool initialized:", wiki_tool)  # Debug print
    
    return question_router, wiki_tool

def main():
    st.title("ResearcherAI")

    # Initialize session state
    if 'vector_store' not in st.session_state:
        st.session_state['vector_store'] = None
    if 'astra_connected' not in st.session_state:
        st.session_state['astra_connected'] = False

    # Initialize services
    services = initialize_services()
    if not services:
        return
    question_router, wiki_tool = services

    # AstraDB Connection
    ASTRA_DB_APPLICATION_TOKEN = st.text_input("Enter AstraDB Token:", type="password")
    ASTRA_DB_ID = st.text_input("Enter AstraDB ID:")

    if st.button("Connect to AstraDB"):
        success = init_astradb_connection(ASTRA_DB_APPLICATION_TOKEN, ASTRA_DB_ID)
        if success:
            st.session_state['astra_connected'] = True
            st.success("Connected to AstraDB successfully!")
        else:
            st.error("Failed to connect to AstraDB")

    # File Upload
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file is not None:
        try:
            # Save and process PDF
            file_path = f"temp_{uploaded_file.name}"
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Process document
            doc_splits = process_pdf(file_path)
            if doc_splits:
                # Create vector store
                vector_store = create_vector_store(doc_splits)
                st.write(f"Inserted {len(doc_splits)} documents.")
                st.session_state['vector_store'] = vector_store

                # Create workflow
                workflow = create_workflow()

                # Query interface
                query = st.text_input("Enter your query:")
                if st.button("Search"):
                    if not wiki_tool:
                        st.error("Wiki tool not initialized properly")
                        return

                    # Initialize state
                    state = {
                        "question": query,
                        "generation": "",
                        "documents": [],
                        "router": question_router,
                        "wiki_tool": wiki_tool,  # Verify this is not None
                        "retriever": vector_store.as_retriever()
                    }

                    # Debug print
                    print("State initialized with:", state.keys())

                    # Execute workflow
                    try:
                        with st.spinner("Processing query..."):
                            result = workflow.invoke(state)
                    except Exception as e:
                        st.error(f"Workflow execution failed: {e}")
                        return

                    # Display results
                    st.write("Search Results:")
                    if isinstance(result["documents"], list):
                        for doc in result["documents"]:
                            st.write(doc.page_content)
                    else:
                        st.write(result["documents"].page_content)

            # Cleanup
            os.remove(file_path)

        except Exception as e:
            st.error(f"Error processing file: {e}")

if __name__ == "__main__":
    main()