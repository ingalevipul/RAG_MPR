# 🛡️ SupplyGuard — AI Supply Chain Risk Assessment

A complete, end-to-end **Generative AI + RAG system** for proactive supply chain risk assessment using **LangChain**, **Gemini**, and **ChromaDB**.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA INGESTION LAYER                      │
│  RSS Feeds (Reuters/BBC) │ OpenWeatherMap │ Historical DB    │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                    VECTOR STORE (ChromaDB)                   │
│      HuggingFace Embeddings (sentence-transformers)         │
│      all-MiniLM-L6-v2  │  Local / Free  │  Persistent       │
└───────────────────────┬─────────────────────────────────────┘
                        │  RAG Retrieval
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                    LANGCHAIN RAG CHAINS                      │
│   RiskAssessmentChain │ ScenarioChain │ ChatChain           │
│              Google Gemini 1.5 Flash (FREE tier)            │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                    FASTAPI BACKEND                           │
│   /api/assess-risk │ /api/generate-scenarios │ /api/chat    │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              FRONTEND DASHBOARD (HTML/CSS/JS)                │
│   Risk Dashboard │ Assessment │ Scenarios │ Chat │ Weather   │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔧 Tech Stack (All Free / Open Source)

| Component | Tool | Why |
|-----------|------|-----|
| **LLM** | Google Gemini 1.5 Flash | Free tier: 15 RPM, 1M tokens/day |
| **LLM Framework** | LangChain | Orchestration, prompts, chains |
| **Vector Store** | ChromaDB | Free, local, persistent |
| **Embeddings** | HuggingFace sentence-transformers | Free, local, no API needed |
| **News Data** | RSS Feeds (Reuters, BBC, GDACS) | Completely free |
| **Weather Data** | OpenWeatherMap (optional) | Free tier: 1000 calls/day |
| **Backend** | FastAPI + Uvicorn | Fast, async |
| **Frontend** | Pure HTML/CSS/JS | No framework needed |

---

## 🚀 Quick Start

### 1. Get Free API Keys
- **Gemini** (required): https://aistudio.google.com/apikey  
- **OpenWeatherMap** (optional): https://openweathermap.org/api

### 2. Setup
```bash
# Clone or download the project
cd supply_chain_risk

# Copy and configure environment
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY

# One-command run
python run.py
```

### 3. Open Dashboard
Navigate to: **http://localhost:8000**

---

## 📁 Project Structure

```
supply_chain_risk/
├── backend/
│   ├── __init__.py
│   ├── main.py              # FastAPI server + all API routes
│   ├── data_ingestion.py    # News RSS, Weather, Historical data
│   ├── vector_store.py      # ChromaDB + HuggingFace embeddings
│   └── rag_chain.py         # LangChain chains with Gemini
├── frontend/
│   └── index.html           # Complete dashboard UI
├── vector_store/
│   └── chroma_db/           # Persistent vector DB (auto-created)
├── .env.example             # Environment template
├── requirements.txt         # Python dependencies
├── run.py                   # One-command launcher
└── README.md
```

---

## 🎯 Features

### Dashboard
- Real-time data ingestion stats
- Quick risk assessment with natural language query
- System health monitoring

### Risk Assessment
- Company profile customization (industry, regions, materials)
- Full RAG-powered risk report with:
  - Overall risk score (1-100)
  - Identified risks with severity, probability, time horizon
  - Mitigation strategies with timelines and cost estimates
  - Immediate action items

### Scenario Planning
- Generate 3 what-if scenarios by:
  - Type: Natural Disaster / Geopolitical / Pandemic / Labor / Cyber
  - Region: Asia Pacific / Europe / North America / etc.
  - Industry: Electronics / Automotive / Pharma / etc.
- Each scenario includes probability/severity scores, cascade effects, recovery time, and financial impact

### Weather Monitoring
- Live weather for 8 major ports/logistics hubs
- Severe weather alerts for supply chain cities

### Historical Disruptions
- 10 major historical events database
- Suez Canal, COVID-19, Ukraine War, Taiwan Chips, etc.

### AI Chat
- Conversational interface with RAG
- Ask anything about current risks and historical patterns

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | System status |
| GET | `/api/data-summary` | Data ingestion stats |
| POST | `/api/assess-risk` | Full risk assessment |
| POST | `/api/generate-scenarios` | Scenario generation |
| POST | `/api/chat` | Conversational Q&A |
| GET | `/api/historical-disruptions` | Historical data |
| GET | `/api/current-weather` | Weather data |
| POST | `/api/refresh-data` | Re-fetch + rebuild vector store |

---

## 📋 Example API Usage

```python
import requests

# Risk Assessment
response = requests.post("http://localhost:8000/api/assess-risk", json={
    "query": "What are semiconductor supply chain risks in Q1 2025?",
    "company_profile": {
        "name": "TechCorp",
        "industry": "Electronics",
        "regions": ["Asia Pacific", "Europe"],
        "critical_inputs": ["semiconductors", "rare earth metals"]
    }
})
print(response.json())

# Scenario Generation
response = requests.post("http://localhost:8000/api/generate-scenarios", json={
    "scenario_type": "geopolitical",
    "region": "Asia Pacific",
    "industry": "Automotive"
})
print(response.json())

# Chat
response = requests.post("http://localhost:8000/api/chat", json={
    "message": "How did the Suez Canal blockage affect global supply chains?"
})
print(response.json()["answer"])
```

---

## ⚙️ How RAG Works in This System

1. **Ingest**: News RSS → Weather API → Historical DB → Documents
2. **Embed**: Each document converted to vector using `all-MiniLM-L6-v2` (local, free)
3. **Store**: Vectors stored in ChromaDB on disk
4. **Query**: User query → embed → similarity search → top-8 relevant docs retrieved
5. **Generate**: Retrieved context + user query → Gemini prompt → structured JSON output
6. **Parse**: JSON parsed and returned to frontend

---

## 🆓 Staying Within Free Tiers

- **Gemini 1.5 Flash**: 15 requests/minute, 1M tokens/day — more than enough
- **Embeddings**: Run locally, zero cost
- **ChromaDB**: Local storage, zero cost  
- **News RSS**: No rate limits, completely free
- **Weather API**: 1000 calls/day on free tier (optional)

---

## 🔮 Potential Extensions

- Add more data sources: GDELT, UN ReliefWeb API, Bloomberg RSS
- Add email/Slack alerts for high-severity risks
- Add PostgreSQL for persistent assessment history
- Add user authentication (FastAPI JWT)
- Deploy to Google Cloud Run (has free tier)
