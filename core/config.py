import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

def setup_environment():
    load_dotenv()
    
    # LangSmith tracing enabled
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "Automated-Financial-Analyst"
    
    print("LLMOps: LangSmith Observability is ENABLED.")
    print(f"Tracing to LangSmith Project: '{os.environ.get('LANGCHAIN_PROJECT')}'")

def initialize_llm():
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is missing. Please check your .env file.")
    
    primary_llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        model="google/gemma-3-4b-it:free",
        default_headers={"HTTP-Referer": "http://localhost", "X-Title": "Autonomous Financial Analyst"}
    )

    backup_llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        model="openrouter/free", 
        default_headers={"HTTP-Referer": "http://localhost", "X-Title": "Autonomous Financial Analyst"}
    )
    
    robust_llm = primary_llm.with_fallbacks([backup_llm])
    
    return robust_llm