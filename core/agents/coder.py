import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_experimental.tools import PythonREPLTool

def initialize_llm():
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is missing. Please check your .env file.")
    
    # Primary Model
    primary_llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        model="google/gemma-3-4b-it:free",
        default_headers={"HTTP-Referer": "http://localhost", "X-Title": "Autonomous Financial Analyst"}
    )

    # Backup with free router
    backup_llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        model="openrouter/free", 
        default_headers={"HTTP-Referer": "http://localhost", "X-Title": "Autonomous Financial Analyst"}
    )
    
    robust_llm = primary_llm.with_fallbacks([backup_llm])
    
    return robust_llm

def run_coder_agent():
    print("Initializing Coder Agent...")
    
    try:
        llm = initialize_llm()
        
        python_tool = PythonREPLTool()
        tools = [python_tool]
        
        template = """Answer the following questions as best you can. You have access to the following tools:

            {tools}

            Use the following format:

            Question: the input question you must answer
            Thought: you should always think about what to do
            Action: the action to take, should be one of [{tool_names}]
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I now know the final answer
            Final Answer: the final answer to the original input question

            Begin!

            Question: {input}
            Thought:{agent_scratchpad}"""
        
        prompt = PromptTemplate.from_template(template)
        
        agent = create_react_agent(llm, tools, prompt)
        
        agent_executor = AgentExecutor(
            agent=agent, 
            tools=tools, 
            verbose=True, 
            handle_parsing_errors=True,
            max_iterations=3,
            early_stopping_method="generate"
        )
        
        task_prompt = (
            "You are an expert Data Scientist. Your task is to write and execute Python code "
            "to generate data visualizations. You have access to a Python REPL. Write a script "
            "using matplotlib to plot the following mock quarterly revenue for Tesla "
            "(Q1: $21B, Q2: $24B, Q3: $23B, Q4: $25B). Save the chart as revenue_chart.png "
            "in the current directory. Only return the final success message once the image is saved."
        )
        
        print("Sending visualization task to the agent...\n")
        
        response = agent_executor.invoke({"input": task_prompt})
        
        print("\n--- Coder Agent Output ---")
        print(response.get("output", "No output returned."))
        print("----------------------------")
        
    except Exception as e:
        print(f"\nPipeline Error: {e}")

if __name__ == "__main__":
    run_coder_agent()