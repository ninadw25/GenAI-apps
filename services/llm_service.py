from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from models.route_query import RouteQuery

def init_llm(api_key: str):
    """Initialize LLM with API key."""
    llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.3-70b-versatile")
    return llm.with_structured_output(RouteQuery)

def create_router(llm):
    """Create question router with prompt."""
    system = """You are an expert at routing a user question to a vectorstore or wikipedia.
    The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
    Use the vectorstore for questions on these topics. Otherwise, use wiki-search."""
    
    route_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "{question}"),
    ])
    
    return route_prompt | llm