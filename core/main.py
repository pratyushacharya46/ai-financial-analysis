import os
import sqlite3
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_experimental.tools import PythonREPLTool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

from core.config import initialize_llm

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  
    draft_answer: str
    critic_feedback: str
    score: int
    revision_number: int
    revenue_data: str
    chart_path: str

def researcher_node(state: AgentState):
    print("\n[Manager] : Routing to Junior Analyst (Researcher)...")
    llm = initialize_llm()
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory="core/data/chroma_db", embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    last_user_message = next((m.content for m in reversed(state["messages"]) if m.type == "human"), "")
    docs = retriever.invoke(last_user_message)
    context = "\n\n".join([d.page_content for d in docs])
    
    current_rev = state.get("revision_number", 0)
    
    # Check if this is a Revision loop from the Critic
    if current_rev > 0 and state.get("critic_feedback") and "chart" not in state.get("critic_feedback", "").lower():
        print(f"[Researcher] : Revising draft (Revision #{current_rev + 1}) based on feedback...")
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a Financial Analyst. Improve your previous answer using the Context and Reviewer feedback.\n"
                       "CRITICAL: DO NOT write Python code for charts. Focus ONLY on text and data extraction.\n\n"
                       "Context: {context}\n\nReviewer Feedback: {feedback}"),
            MessagesPlaceholder(variable_name="messages")
        ])
        chain = prompt | llm
        response = chain.invoke({"messages": state["messages"], "context": context, "feedback": state["critic_feedback"]})
        
        return {
            "draft_answer": response.content, 
            "messages": [AIMessage(content=response.content)]
        }
    else:
        # NEW User Query
        print("[Researcher] : Reading 10-K database and drafting initial response...")
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert Financial Analyst. Answer the user's question concisely using ONLY the provided Context.\n"
                       "CRITICAL RULES FOR CHARTS:\n"
                       "1. DO NOT write Python code, matplotlib, or scripts. EVER.\n"
                       "2. If the user asks for a chart/visualization, extract the relevant numerical data from the Context and present it in a clean markdown table.\n"
                       "3. Conclude by saying: 'I have extracted the data. The Data Scientist will now generate the chart.'\n\n"
                       "Context: {context}"),
            MessagesPlaceholder(variable_name="messages")
        ])
        chain = prompt | llm
        response = chain.invoke({"messages": state["messages"], "context": context})

        return {
            "draft_answer": response.content, 
            "messages": [AIMessage(content=response.content)],
            "chart_path": "",      
            "critic_feedback": "", 
            "score": 0
        }

def check_for_chart(state: AgentState):
    last_message = next((m.content for m in reversed(state["messages"]) if m.type == "human"), "").lower()
    
    if any(word in last_message for word in ['plot', 'chart', 'graph', 'visual']):
        return "coder"
    else:
        return "critic"

def coder_node(state: AgentState):
    print("\n[Manager] : Routing to Data Scientist (Coder)...")
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
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    
    analyst_data = state.get("draft_answer", "")
    fallback_data = state.get("revenue_data", "No data provided.")
    last_user_message = next((m.content for m in reversed(state["messages"]) if m.type == "human"), "")
    current_rev = state.get("revision_number", 0)
    
    if current_rev > 0 and state.get("critic_feedback") and "chart" in state.get("critic_feedback", "").lower():
        print(f"[Coder] : Fixing code (Revision #{current_rev + 1}) based on feedback...")
        task_prompt = (
            f"You are an expert Data Scientist. You previously tried to generate a chart but received this feedback:\n"
            f"{state['critic_feedback']}\n\n"
            f"Please write a new Python script to fix the error. Use this data extracted by the Analyst:\n{analyst_data}\n"
            f"If no explicit numbers are present above, use this fallback data: {fallback_data}\n"
            "Save the chart as revenue_chart.png in the current directory. Only return the final success message."
        )
    else:
        print("[Coder] : Writing initial Python script...")
        task_prompt = (
            f"You are an expert Data Scientist. The user asked: '{last_user_message}'.\n"
            f"The Financial Analyst has extracted the following data from the SEC filings:\n{analyst_data}\n\n"
            f"Write a Python script using matplotlib to visualize the data provided by the Analyst. "
            f"If the Analyst did not provide specific numbers, use this fallback data: {fallback_data}\n"
            "Save the chart exactly as 'revenue_chart.png' in the current directory. "
            "Only return the final success message once the image is saved."
        )
    
    try:
        response = agent_executor.invoke({"input": task_prompt})
        output_msg = response.get("output", "Successfully generated chart.")
        
        return {
            "chart_path": "revenue_chart.png",
            "messages": [AIMessage(content=f"Chart generated successfully: {output_msg}")]
        }
    except Exception as e:
        error_str = str(e)
        print(f"\nCoder Error: {error_str}")
        
        # Abort on API limits
        if "429" in error_str or "402" in error_str or "Rate limit" in error_str:
            raise RuntimeError(f"API Rate Limit: {error_str}") from e
            
        return {
            "chart_path": "Error generating chart",
            "messages": [AIMessage(content=f"Error generating chart: {error_str}")]
        }

def critic_node(state: AgentState):
    print("\n[Manager] : Routing to Senior Reviewer (Critic)...")
    llm = initialize_llm()
    
    last_user_message = next((m.content for m in reversed(state["messages"]) if m.type == "human"), "")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Senior Financial Reviewer. Review the Junior Analyst's text output AND the Data Scientist's chart path (if a chart was requested). "
                   "You MUST provide a final score from 1 to 10 based on accuracy and completeness. "
                   "Format your response EXACTLY like this:\n"
                   "SCORE: [number]\nFEEDBACK: [your detailed critique]"),
        ("human", "Original Request: {question}\n\nAnalyst's Output: {draft_answer}\n\nChart Generated At: {chart_path}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({
        "question": last_user_message, 
        "draft_answer": state.get("draft_answer", "No text provided."),
        "chart_path": state.get("chart_path", "No chart requested or generated.")
    })
    content = response.content
    
    score = 5 
    try:
        if "SCORE:" in content:
            score_str = content.split("SCORE:")[1].split("\n")[0].strip()
            score = int(score_str)
    except Exception:
        print("[Critic] : Could not parse exact score, defaulting to 5.")
        
    print(f"[Critic] : Gave a score of {score}/10")
    
    return {"critic_feedback": content, "score": score, "revision_number": state.get("revision_number", 0) + 1}

def route_evaluation(state: AgentState):
    score = state.get("score", 0)
    revisions = state.get("revision_number", 0)
    
    if score >= 8:
        print("\n[Manager] : Output approved by Critic! Finalizing report...")
        return END
    elif revisions >= 3:
        print("\n[Manager] : Maximum revisions (3) reached. Approving best-effort report...")
        return END
    else:
        feedback = state.get("critic_feedback", "").lower()
        if any(word in feedback for word in ['chart', 'plot', 'image', 'graph', 'matplotlib', 'coder']):
            print("\n[Manager] : Chart needs fixing. Routing back to Coder...")
            return "coder"
        else:
            print("\n[Manager] : Text needs improvement. Routing back to Researcher...")
            return "researcher"

def build_graph():
    builder = StateGraph(AgentState)
    
    builder.add_node("researcher", researcher_node)
    builder.add_node("coder", coder_node)
    builder.add_node("critic", critic_node)
    
    builder.add_edge(START, "researcher")
    builder.add_conditional_edges("researcher", check_for_chart)
    builder.add_edge("coder", "critic")
    
    builder.add_conditional_edges("critic", route_evaluation)
    
    os.makedirs("core/data", exist_ok=True)
    conn = sqlite3.connect("core/data/checkpoints.sqlite", check_same_thread=False)
    memory = SqliteSaver(conn)
    return builder.compile(checkpointer=memory)

if __name__ == "__main__":
    pass