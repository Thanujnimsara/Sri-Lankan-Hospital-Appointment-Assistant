import streamlit as st
import os

st.set_page_config(
    page_title="SL Hospital Assistant",
    page_icon="🏥",
    layout="centered"
)

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
</style>
""", unsafe_allow_html=True)

# API Keys from Streamlit Secrets or Environment
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
if "OPENROUTER_API_KEY" in st.secrets:
    os.environ["OPENROUTER_API_KEY"] = st.secrets["OPENROUTER_API_KEY"]

# Sidebar for configuration
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/hospital.png", width=70)
    st.title("Settings & Status")
    st.markdown("---")
    
    groq_key = st.text_input("GROQ API Key", value=os.environ.get("GROQ_API_KEY", ""), type="password")
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key
        
    openrouter_key = st.text_input("OpenRouter / OpenAI API Key", value=os.environ.get("OPENROUTER_API_KEY", ""), type="password")
    if openrouter_key:
        os.environ["OPENROUTER_API_KEY"] = openrouter_key
        
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
    st.caption("🇱🇰 Sri Lanka Healthcare AI Assistant | Multi-Agent RAG Architecture")

# Main Header UI
st.markdown("""
<div class="header-box">
    <div class="header-title">🏥 Sri Lankan Hospital & Appointment Assistant</div>
    <div class="header-sub">Find specialists, check hospital guidelines, and navigate eChannelling/Doc990 procedures with AI.</div>
    <div class="badge-container">
        <span class="badge">🤖 Intent Router (Groq Llama 3.1)</span>
        <span class="badge">📚 Vector RAG (ChromaDB)</span>
        <span class="badge">✨ Reflection Synthesizer (OpenRouter)</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="emergency-banner">
    🚨 <b>Emergency Notice:</b> For acute medical emergencies, call free ambulance <b>Suwa Seriya at 1990</b> immediately.
</div>
""", unsafe_allow_html=True)

# Import agent runner dynamically after env key check
try:
    from src.agents import run_assistant
    agents_ready = True
except Exception as err:
    agents_ready = False
    st.warning(f"Note: Agent components loading or key setup required ({err})")

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
            response = run_assistant(prompt)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.chat_message("assistant").write(response)
        except Exception as e:
            st.error(f"Error executing agent pipeline: {e}")
