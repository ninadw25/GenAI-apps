from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langchain.schema import Document

def init_wiki_search():
    """Initialize Wikipedia search wrapper."""
    try:
        api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
        wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)
        return wiki_tool
    except Exception as e:
        print(f"Error initializing wiki search: {e}")
        return None
    

def perform_wiki_search(wiki_tool, query):
    """Perform Wikipedia search."""
    results = wiki_tool.invoke({"query": query})
    return Document(page_content=results)