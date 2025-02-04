from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from typing import List, Any
from config.settings import EMBEDDING_MODEL, VECTOR_STORE_TABLE, CONSISTENCY_LEVEL

def create_vector_store(documents: List) -> Cassandra:
    """Create and populate vector store with documents."""
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        
        vector_store = Cassandra(
            embedding=embeddings,
            table_name=VECTOR_STORE_TABLE,
            session=None,
            keyspace=None
        )
        
        vector_store.add_documents(documents)
        return vector_store
    except Exception as e:
        print(f"Error creating vector store: {e}")
        raise

def perform_search(vector_store: Cassandra, query: str) -> Any:
    """Perform search query on vector store."""
    try:
        retriever = vector_store.as_retriever()
        return retriever.invoke(query, ConsistencyLevel=CONSISTENCY_LEVEL)
    except Exception as e:
        print(f"Error performing search: {e}")
        raise