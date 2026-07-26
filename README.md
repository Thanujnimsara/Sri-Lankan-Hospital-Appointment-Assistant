# 🏥 Sri Lankan Hospital Appointment Assistant

An intelligent, multi-agent Retrieval-Augmented Generation (RAG) assistant designed to simplify medical appointments, specialist matching, hospital navigation, and channelling service procedures (eChannelling, Doc990) across Sri Lanka.

---

## 📌 Project Overview
Navigating the healthcare system in Sri Lanka—whether finding the appropriate medical specialist, understanding booking policies across private providers (Asiri, Lanka Hospitals, Nawaloka, Durdans, Hemas, Suwasewana), or learning government hospital (NHSL) referral procedures—can be confusing for patients.

This project provides an AI-powered appointment assistant utilizing **LangGraph Multi-Agent Workflows**, **ChromaDB Vector Retrieval**, **Groq Fast Routing**, and **OpenRouter Synthesis** to deliver precise, context-aware guidance with built-in emergency disclaimers.

---

## 🏗️ Agentic Design Patterns Used

1. **Router Pattern (`src/agents.py` - `router_agent`)**:
   - Classifies incoming user queries into discrete intent categories (`APPOINTMENT_PROCEDURE`, `SPECIALIST_MATCH`, `HOSPITAL_INFO`) using ultra-fast Groq Llama 3.1 8B inference.
2. **Tool-Use / RAG Pattern (`src/rag_pipeline.py` & `rag_retriever_agent`)**:
   - Performs semantic similarity search against a 20-file local ChromaDB vector store containing domain-specific Sri Lankan medical, hospital, and channelling knowledge.
3. **Reflection & Guardrail Pattern (`src/agents.py` - `synthesis_agent`)**:
   - Synthesizes retrieved context with the user query, formats structured recommendations, and enforces safety guardrails by appending mandatory Suwa Seriya 1990 emergency disclaimers.

---

## 🔄 Agent-to-Agent Communication Diagram

```mermaid
graph TD
    User([👤 User Query]) --> Router["🤖 Intent Router Agent\n(Groq Llama 3.1 8B)"]
    Router -->|Intent Tag| RAG["📚 RAG Retriever Agent\n(ChromaDB VectorStore)"]
    RAG -->|Top-K Context Chunks| Synthesizer["✨ Reflection & Synthesis Agent\n(OpenRouter / GPT-4o-mini)"]
    Synthesizer -->|Structured Response + Safety Disclaimer| Output([💬 Streamlit UI / User])
```

---

## 📊 Model Selection Justification Table

| Sub-task | Model (Provider) | Why Chosen |
| :--- | :--- | :--- |
| **Intent Classification** | `llama-3.1-8b-instant` (Groq) | Sub-100ms ultra-low latency, high throughput, zero token cost for fast routing. |
| **Final Synthesis & Advice** | `openai/gpt-4o-mini` / `claude-3.5-sonnet` (OpenRouter) | Superior reasoning, context synthesis, high safety compliance, clear formatting. |

---

## 🧪 5-Query RAG Evaluation Table

| Query | Retrieved Document | Relevant? | Comment |
| :--- | :--- | :--- | :--- |
| **How to book via eChannelling?** | `08_echannelling_faq.txt` | Yes | Retrieved step-by-step payment and booking guide. |
| **Who to see for chest pain?** | `12_specialist_cardiology.txt` | Yes | Matched cardiologist & emergency caution for heart symptoms. |
| **Nawaloka OPD hours?** | `03_nawaloka_hospitals.txt` | Yes | Returned accurate 24/7 OPD timings and specialist clinic hours. |
| **What is 1990 number?** | `17_emergency_1990.txt` | Yes | Correctly retrieved Suwa Seriya free ambulance hotline details. |
| **eZ Cash payment for appointments?** | `10_payment_methods.txt` | Yes | Accurately detailed mobile wallet and add-to-bill payment steps. |

---

## 📁 Repository Structure

```
sri-lankan-hospital-assistant/
├── .gitignore
├── README.md
├── requirements.txt
├── app.py
├── data/
│   ├── 01_asiri_hospitals.txt
│   ├── 02_lanka_hospitals.txt
│   ├── 03_nawaloka_hospitals.txt
│   ├── 04_durdans_hospitals.txt
│   ├── 05_hemas_hospitals.txt
│   ├── 06_suwasewana_kandy.txt
│   ├── 07_nhsl_public.txt
│   ├── 08_echannelling_faq.txt
│   ├── 09_doc990_faq.txt
│   ├── 10_payment_methods.txt
│   ├── 11_specialist_vp.txt
│   ├── 12_specialist_cardiology.txt
│   ├── 13_specialist_dermatology.txt
│   ├── 14_specialist_neurology.txt
│   ├── 15_specialist_orthopedics.txt
│   ├── 16_specialist_ent.txt
│   ├── 17_emergency_1990.txt
│   ├── 18_channelling_cancellation.txt
│   ├── 19_senior_citizen_discounts.txt
│   └── 20_lab_reports_collection.txt
└── src/
    ├── __init__.py
    ├── rag_pipeline.py
    └── agents.py
```

---

## 🚀 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd sri-lankan-hospital-assistant
   ```

2. **Set up Virtual Environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Environment Variables (or set in sidebar / `.streamlit/secrets.toml`):**
   ```bash
   export GROQ_API_KEY="your-groq-api-key"
   export OPENROUTER_API_KEY="your-openrouter-api-key"
   ```

5. **Launch Streamlit App:**
   ```bash
   streamlit run app.py
   ```

---

## ☁️ Streamlit Cloud Deployment Instructions

1. Push your code to a public repository on **GitHub**.
2. Go to **[share.streamlit.io](https://share.streamlit.io)** and log in with your GitHub account.
3. Select your repository, branch (`main`), and set Main file path to `app.py`.
4. In **Advanced Settings -> Secrets**, enter your keys:
   ```toml
   GROQ_API_KEY = "gsk_..."
   OPENROUTER_API_KEY = "sk-or-v1-..."
   ```
5. Click **Deploy!**
