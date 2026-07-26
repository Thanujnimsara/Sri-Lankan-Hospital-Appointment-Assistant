import os
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from src.rag_pipeline import query_rag
from src.database import save_chat_log

# State Definition
class AgentState(TypedDict):
    user_query: str
    intent: str
    retrieved_context: str
    final_response: str

def get_router_llm():
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key or api_key.startswith("gsk_...") or api_key == "your-groq-api-key":
        raise ValueError("GROQ_API_KEY is not set or invalid. Please enter a valid Groq API key (starts with 'gsk_') in the sidebar under Settings.")
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        api_key=api_key
    )

def get_synthesis_llm():
    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    
    # If valid OpenRouter or OpenAI key exists
    if openrouter_key and not openrouter_key.startswith("sk-or-v1-...") and openrouter_key != "your-openrouter-api-key":
        base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        return ChatOpenAI(
            model="openai/gpt-4o-mini",
            openai_api_base=base_url,
            openai_api_key=openrouter_key
        )
    elif openai_key:
        return ChatOpenAI(
            model="gpt-4o-mini",
            openai_api_key=openai_key
        )
    else:
        # Fallback to Groq for synthesis if OpenRouter key is not provided
        return get_router_llm()

# Node 1: Intent Router Agent
def router_agent(state: AgentState) -> AgentState:
    query = state["user_query"]
    prompt = f"""Classify the user input into ONE of these categories:
    - 'APPOINTMENT_PROCEDURE': questions about eChannelling, Doc990, payments, refunds.
    - 'SPECIALIST_MATCH': symptoms or finding the right doctor specialty.
    - 'HOSPITAL_INFO': general hospital locations, OPD, emergency.
    
    Query: {query}
    Return ONLY the category name."""
    
    router_llm = get_router_llm()
    res = router_llm.invoke(prompt)
    state["intent"] = res.content.strip()
    return state

# Node 2: RAG Retrieval Agent
def rag_retriever_agent(state: AgentState) -> AgentState:
    context = query_rag(state["user_query"], k=3)
    state["retrieved_context"] = context
    return state

# Node 3: Synthesis & Reflection Agent
def synthesis_agent(state: AgentState) -> AgentState:
    prompt = f"""You are the Sri Lankan Hospital Appointment Assistant.
    Intent Identified: {state['intent']}
    Context from Knowledge Base:
    {state['retrieved_context']}
    
    User Query: {state['user_query']}
    
    Provide a helpful, structured, and polite response. 
    ALWAYS include a disclaimer: 'Note: For medical emergencies, call Suwa Seriya at 1990 immediately.'"""
    
    synthesis_llm = get_synthesis_llm()
    res = synthesis_llm.invoke(prompt)
    state["final_response"] = res.content
    return state

# Construct Graph
workflow = StateGraph(AgentState)

workflow.add_node("router", router_agent)
workflow.add_node("rag", rag_retriever_agent)
workflow.add_node("synthesizer", synthesis_agent)

workflow.set_entry_point("router")
workflow.add_edge("router", "rag")
workflow.add_edge("rag", "synthesizer")
workflow.add_edge("synthesizer", END)

app_graph = workflow.compile()

def run_assistant(query: str, session_id: str = "default_session"):
    initial_state = {
        "user_query": query,
        "intent": "",
        "retrieved_context": "",
        "final_response": ""
    }
    output = app_graph.invoke(initial_state)
    
    # Save log to MongoDB database
    save_chat_log(
        session_id=session_id,
        user_query=query,
        intent=output.get("intent", "UNKNOWN"),
        retrieved_context=output.get("retrieved_context", ""),
        response=output.get("final_response", "")
    )
    
    return output["final_response"]
