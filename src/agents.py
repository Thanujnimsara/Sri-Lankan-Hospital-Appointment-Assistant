import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Ensure .env is always loaded with override regardless of working directory
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)
load_dotenv(override=True)

from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from src.rag_pipeline import query_rag
from src.database import save_chat_log

def clean_utf8(text) -> str:
    if text is None:
        return ""
    if isinstance(text, bytes):
        return text.decode("utf-8", errors="ignore")
    return str(text)

# State Definition
class AgentState(TypedDict):
    user_query: str
    intent: str
    retrieved_context: str
    final_response: str

def is_valid_key(key: str) -> bool:
    if not key:
        return False
    k = key.strip().lower()
    return not (k.startswith("gsk_...") or k.startswith("sk-or-v1-...") or "your_" in k or "your-" in k or len(k) < 20)

def get_router_llm():
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not is_valid_key(api_key):
        raise ValueError("GROQ_API_KEY is not set or invalid. Please configure GROQ_API_KEY in your .env file.")
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        api_key=api_key
    )

def get_synthesis_llm():
    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    
    if is_valid_key(openrouter_key):
        base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        return ChatOpenAI(
            model="openai/gpt-4o-mini",
            openai_api_base=base_url,
            openai_api_key=openrouter_key
        )
    elif is_valid_key(openai_key):
        return ChatOpenAI(
            model="gpt-4o-mini",
            openai_api_key=openai_key
        )
    else:
        return get_router_llm()

# Node 1: Intent Router Agent
def router_agent(state: AgentState) -> AgentState:
    query = clean_utf8(state["user_query"])
    prompt = f"""Classify the user input into ONE of these categories:
    - 'APPOINTMENT_PROCEDURE': questions about eChannelling, Doc990, payments, refunds.
    - 'SPECIALIST_MATCH': symptoms or finding the right doctor specialty.
    - 'HOSPITAL_INFO': general hospital locations, OPD, emergency.
    
    Query: {query}
    Return ONLY the category name."""
    
    router_llm = get_router_llm()
    res = router_llm.invoke(prompt)
    state["intent"] = clean_utf8(res.content.strip())
    return state

# Node 2: RAG Retrieval Agent
def rag_retriever_agent(state: AgentState) -> AgentState:
    context = clean_utf8(query_rag(state["user_query"], k=3))
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
    state["final_response"] = clean_utf8(res.content)
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
    query = clean_utf8(query)
    initial_state = {
        "user_query": query,
        "intent": "HOSPITAL_INFO",
        "retrieved_context": "",
        "final_response": ""
    }
    
    try:
        output = app_graph.invoke(initial_state)
        final_res = clean_utf8(output.get("final_response", ""))
        intent_tag = clean_utf8(output.get("intent", "HOSPITAL_INFO"))
        retrieved_ctx = clean_utf8(output.get("retrieved_context", ""))
    except Exception as e:
        err_msg = str(e)
        if "GROQ_API_KEY" in err_msg or "401" in err_msg or "invalid_api_key" in err_msg:
            raise e
        # Safe direct invocation fallback
        intent_tag = "HOSPITAL_INFO"
        retrieved_ctx = clean_utf8(query_rag(query, k=3))
        prompt = f"Context:\n{retrieved_ctx}\n\nUser Query: {query}\nProvide a polite response. Always state: Note: For medical emergencies, call Suwa Seriya at 1990 immediately."
        llm = get_router_llm()
        res = llm.invoke(prompt)
        final_res = clean_utf8(res.content)
    
    # Save log to MongoDB database
    save_chat_log(
        session_id=session_id,
        user_query=query,
        intent=intent_tag,
        retrieved_context=retrieved_ctx,
        response=final_res
    )
    
    return final_res
