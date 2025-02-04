from langgraph.graph import END, StateGraph, START
from models.graph_state import GraphState
from graph.nodes import wiki_search, retrieve, route_question

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