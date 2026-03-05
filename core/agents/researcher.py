import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

def initialize_llm():
    """Sets up the OpenRouter connection with automatic fallbacks."""
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
    
    return primary_llm.with_fallbacks([backup_llm])

def run_researcher_agent():
    print("Initializing Researcher Agent (RAG Mode)...")
    
    try:
        llm = initialize_llm()
        
        print("Loading local embedding model...")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        db_dir = "core/data/chroma_db"
        if not os.path.exists(db_dir):
            raise FileNotFoundError("ChromaDB not found! Please run tools/vector_store.py first.")
            
        print("Connecting to local Vector Database...")
        vectorstore = Chroma(persist_directory=db_dir, embedding_function=embeddings)
        
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        
        system_prompt = (
            "You are an expert Financial Analyst. Use the following pieces of retrieved "
            "context from a company's 10-K report to answer the question. "
            "If you don't know the answer based on the context, say that you don't know. "
            "Keep the answer concise and professional.\n\n"
            "Context: {context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        question = "What are the primary risk factors related to Tesla's supply chain and manufacturing?"
        
        print(f"\nSearching database for: '{question}'...")
        response = rag_chain.invoke({"input": question})

        print("\n--- Researcher Agent Analysis ---")
        print(response["answer"])
        print("\n📚 Sources used:")
        for i, doc in enumerate(response["context"]):
            print(f"  - Document Chunk {i+1} (Length: {len(doc.page_content)} chars)")
        print("-----------------------------------")
        
    except Exception as e:
        print(f"\nPipeline Error: {e}")

if __name__ == "__main__":
    run_researcher_agent()