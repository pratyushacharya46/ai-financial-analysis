import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

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
    
    return primary_llm.with_fallbacks([backup_llm])

def run_critic_agent():
    """Executes the Critic Agent to review the work of other agents."""
    print("🤖 Initializing Critic Agent (Senior Reviewer)...")
    
    try:
        llm = initialize_llm()
        
        system_prompt = (
            "You are a Senior Financial Reviewer at a top-tier investment firm. "
            "Your job is to review the output provided by a Junior Financial Analyst (Researcher) "
            "or a Data Scientist (Coder). "
            "Check for accuracy, clarity, and completeness against the original user request. "
            "Provide a brief critique, point out any missing information, and give a final score out of 10. "
            "If the output is flawed, explicitly state what needs to be fixed."
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Original User Request: {request}\n\nAgent Output to Review:\n{output}"),
        ])
        
        chain = prompt | llm
        
        test_request = "What are the primary risk factors related to Tesla's supply chain and manufacturing?"
        test_output = (
            "1. Supplier Dependency and Shortages\n"
            "2. Demand Forecasting Challenges\n"
            "3. Tariffs and Trade Policies\n"
            "4. Financing Constraints\n"
            "5. Production Scaling Complexities\n"
            "6. Logistical and Cyber Risks"
        )
        
        print("\nReviewing the Junior Analyst's work...")
        response = chain.invoke({
            "request": test_request, 
            "output": test_output
        })
        
        print("\n--- Critic Agent Review ---")
        print(response.content)
        print("-----------------------------")
        
    except Exception as e:
        print(f"\nPipeline Error: {e}")

if __name__ == "__main__":
    run_critic_agent()