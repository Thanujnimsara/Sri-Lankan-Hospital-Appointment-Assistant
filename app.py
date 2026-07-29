import streamlit as st
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Force UTF-8 I/O encoding & disable progress bar output for Windows compatibility
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"
os.environ["TQDM_DISABLE"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

for _stream_name in ("stdout", "stderr"):
    _stream = getattr(sys, _stream_name, None)
    if _stream is not None:
        if hasattr(_stream, "reconfigure"):
            try:
                _stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

# Ensure .env is always loaded with override regardless of working directory
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path, override=True, encoding="utf-8")
load_dotenv(override=True, encoding="utf-8")

import uuid
from src.database import get_database_status, get_recent_logs, get_intent_analytics

st.set_page_config(
    page_title="SL Hospital Assistant",
    page_icon="🏥",
    layout="centered"
)

# Initialize Session ID
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

# Custom CSS for rich aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        color: #f8fafc;
    }
    .header-box {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    .header-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    .header-sub {
        color: #94a3b8;
        font-size: 1.05rem;
    }
    .badge-container {
        display: flex;
        justify-content: center;
        gap: 10px;
        margin-top: 14px;
        flex-wrap: wrap;
    }
    .badge {
        background: rgba(99, 102, 241, 0.15);
        color: #a5b4fc;
        border: 1px solid rgba(129, 140, 248, 0.3);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 500;
    }
    .stChatMessage {
        background: rgba(30, 41, 59, 0.7) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 14px !important;
        margin-bottom: 12px !important;
    }
    .emergency-banner {
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid rgba(239, 68, 68, 0.3);
        color: #fca5a5;
        padding: 10px 16px;
        border-radius: 12px;
        font-size: 0.9rem;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    /* Hide Streamlit password reveal eye icon button */
    button[aria-label="Show password text"],
    button[aria-label="Hide password text"],
    button[title="Show password text"],
    button[title="Hide password text"] {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to filter placeholder keys
def is_real_key(val):
    if not val:
        return False
    val = val.strip()
    return not (val.startswith("gsk_...") or val.startswith("sk-or-v1-...") or "your-" in val or len(val) < 15)

# Load secrets if valid
if "GROQ_API_KEY" in st.secrets and is_real_key(st.secrets["GROQ_API_KEY"]):
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
if "OPENROUTER_API_KEY" in st.secrets and is_real_key(st.secrets["OPENROUTER_API_KEY"]):
    os.environ["OPENROUTER_API_KEY"] = st.secrets["OPENROUTER_API_KEY"]

# Sidebar for configuration & DB Status
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/hospital.png", width=70)
    st.title("Settings & Status")
    st.caption(f"Session ID: `{st.session_state.session_id}`")
    st.markdown("---")
    
    # MongoDB Status Indicator
    db_connected, db_msg = get_database_status()
    if db_connected:
        st.success(f"🍃 {db_msg}")
    else:
        st.warning(f"⚠️ {db_msg}")
        
        
    st.markdown("---")
    st.subheader("💡 Sample Queries")
    st.markdown("""
    - *How do I book a cardiologist at Asiri via eChannelling?*
    - *Who should I consult for severe chest pain?*
    - *What are NHSL public OPD hours and referral steps?*
    - *What is the refund policy for Doc990 cancellations?*
    - *How to pay via eZ Cash or Dialog add-to-bill?*
    """)
    st.markdown("---")
    st.caption("🇱🇰 Sri Lanka Healthcare AI Assistant | Multi-Agent RAG + MongoDB Architecture")

# Main Header UI
st.markdown("""
<div class="header-box">
    <div class="header-title">🏥 Sri Lankan Hospital & Appointment Assistant</div>
    <div class="header-sub">Find specialists, check hospital guidelines, and navigate eChannelling/Doc990 procedures with AI.</div>
    <div class="badge-container">
        <span class="badge">🤖 Intent Router (Groq Llama 3.1)</span>
        <span class="badge">📚 Vector RAG (ChromaDB)</span>
        <span class="badge">🍃 Database (MongoDB)</span>
        <span class="badge">✨ Reflection Synthesizer (Groq/OpenRouter)</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Tabs for Chat vs Database Analytics
tab_chat, tab_analytics = st.tabs(["💬 Assistant Chat", "📊 MongoDB Database & Analytics"])

with tab_chat:
    st.markdown("""
    <div class="emergency-banner">
        🚨 <b>Emergency Notice:</b> For acute medical emergencies, call free ambulance <b>Suwa Seriya at 1990</b> immediately.
    </div>
    """, unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Ayubowan! How can I assist you with your medical appointments, hospital information, or specialist search in Sri Lanka today?"}
        ]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("e.g. How do I book a cardiologist at Asiri via eChannelling?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        with st.spinner("Consulting Sri Lankan hospital knowledge base & agents..."):
            try:
                from src.agents import run_assistant
                response = run_assistant(prompt, session_id=st.session_state.session_id)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.chat_message("assistant").write(response)
            except Exception as e:
                import traceback
                tb_str = traceback.format_exc()
                try:
                    sys.stdout.write("=== DETAILED TRACEBACK ===\n" + tb_str + "\n==========================\n")
                except Exception:
                    pass
                st.error(f"⚠️ **Assistant Error:** {str(e)}")
                st.code(tb_str)

with tab_analytics:
    st.subheader("🍃 MongoDB Persistent Chat Logs & Analytics")
    
    analytics = get_intent_analytics()
    if analytics:
        st.markdown("#### Intent Distribution")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Appointment Procedures", analytics.get("APPOINTMENT_PROCEDURE", 0))
        with col2:
            st.metric("Specialist Match", analytics.get("SPECIALIST_MATCH", 0))
        with col3:
            st.metric("Hospital Info", analytics.get("HOSPITAL_INFO", 0))
    
    st.markdown("---")
    st.markdown("#### Recent Database Logs")
    logs = get_recent_logs(limit=10)
    if logs:
        for log in logs:
            with st.expander(f"🕒 {log.get('timestamp')} | Intent: {log.get('intent')} | Query: {log.get('user_query')[:40]}..."):
                st.write(f"**Session ID:** `{log.get('session_id')}`")
                st.write(f"**User Query:** {log.get('user_query')}")
                st.write(f"**Intent Tag:** `{log.get('intent')}`")
                st.write(f"**Retrieved Context Snippet:**\n```\n{log.get('retrieved_context')[:300]}...\n```")
                st.write(f"**Final AI Response:**\n{log.get('final_response')}")
    else:
        st.info("No logs currently recorded in MongoDB (or MongoDB container is starting up). Submit a chat query to record your first log!")
