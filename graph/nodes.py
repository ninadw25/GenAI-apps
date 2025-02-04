from langchain.schema import Document
from typing import Dict, Any
from langgraph.graph import END, StateGraph, START
from models.graph_state import GraphState

def retrieve(state: Dict[str, Any]) -> Dict[str, Any]:
    """Retrieve documents from vector store."""
    print("---RETRIEVE---")
    question = state["question"]
    documents = state["retriever"].invoke(question)
    return {"documents": documents, "question": question}

def wiki_search(state: Dict[str, Any]) -> Dict[str, Any]:
    """Search Wikipedia using the wiki tool."""
    print("---WIKI SEARCH---")
    if "wiki_tool" not in state:
        raise KeyError("wiki_tool not found in state")
    question = state["question"]
    result = state["wiki_tool"].invoke(question)
    return {"documents": result, "question": question}

def route_question(state: Dict[str, Any]) -> str:
    """Route the question to appropriate search method."""
    print("---ROUTING---")
    question = state["question"].lower()
    
    wikipedia_keywords = ["who", "what", "when", "where", "why", "how"]
    if any(keyword in question for keyword in wikipedia_keywords):
        return "wiki_search"
    return "vectorstore"

def create_workflow():
    """Create and configure the workflow graph."""
    workflow = StateGraph(GraphState)
    
    # Define the nodes
    workflow.add_node("wiki_search", wiki_search)
    workflow.add_node("retrieve", retrieve)
    
    # Build graph
    workflow.add_conditional_edges(
        START,
        route_question,
        {
            "wiki_search": "wiki_search",
            "vectorstore": "retrieve",
        },
    )
    
    workflow.add_edge("retrieve", END)
    workflow.add_edge("wiki_search", END)
    
    return workflow.compile()