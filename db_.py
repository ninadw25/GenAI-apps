import streamlit as st
from utils.database import init_astradb_connection
from utils.document_processor import process_pdf
from services.vector_store import create_vector_store, perform_search
import os

def main():
    st.title("LangGraph with AstraDB - Streamlit App")

    # AstraDB Connection
    ASTRA_DB_APPLICATION_TOKEN = st.text_input("Enter AstraDB Token:", type="password")
    ASTRA_DB_ID = st.text_input("Enter AstraDB ID:")

#AstraCS:szxpndjLLsgmWZudESRhOrbE:fdbc043ea17ac8c5a9828d62c13ffcb9032ac3e08b172f1550427521272440f7
#3559ba0a-651d-4e35-8160-bbffdfaaf9f2

    if st.button("Connect to AstraDB"):
        success = init_astradb_connection(ASTRA_DB_APPLICATION_TOKEN, ASTRA_DB_ID)
        if success:
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

                # Query interface
                query = st.text_input("Enter your query:")
                if st.button("Search"):
                    results = perform_search(vector_store, query)
                    st.write("Search Results:")
                    st.write(results)

            # Cleanup
            os.remove(file_path)

        except Exception as e:
            st.error(f"Error processing file: {e}")

if __name__ == "__main__":
    main()