from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP

def process_pdf(file_path: str) -> List:
    """Process PDF file and split into chunks."""
    try:
        # Load PDF
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        # Split text
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        return text_splitter.split_documents(docs)
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return []