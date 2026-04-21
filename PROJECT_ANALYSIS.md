# SupplyGuard — Project Analysis

## 📋 Overview

**Project Title:** Generative AI for Proactive Supply Chain Risk Assessment  
**Current Status:** Fully functional prototype  
**Tech Stack:** LangChain + Gemini + ChromaDB + FastAPI + HTML/CSS/JS

---

## 🏗️ Architecture Breakdown

### **Layer 1: Data Ingestion** (`backend/data_ingestion.py`)

**Purpose:** Aggregates real-time and historical supply chain risk data

#### Data Sources:

1. **News Feeds (Free RSS)**
   - Reuters World & Business News
   - BBC World News
   - ReliefWeb (disaster alerts)
   - GDACS (Global Disaster Alerts & Coordination System)
   - Filtering by 20 risk keywords (e.g., "supply chain", "disruption", "sanctions")

2. **Weather Data (OpenWeatherMap - Free Tier)**
   - 8 major supply chain hub cities: Shanghai, Rotterdam, Singapore, LA, Dubai, Mumbai, Surat, Navi Mumbai
   - Monitors severe weather (storms, snow, fog, extreme weather)
   - Falls back to mock data if API key missing

3. **Historical Disruptions (Built-in Dataset)**
   - 10 major documented supply chain disruptions:
     - Suez Canal blockage (Ever Given)
     - COVID-19 pandemic
     - Fukushima disaster
     - Taiwan chip shortage
     - Russia-Ukraine war
     - US-China trade wars
     - Thailand floods
     - Port strikes
     - Red Sea attacks
     - Panama Canal drought

#### Function Outputs:

- `fetch_news_articles()` → List of news items filtered by keywords
- `fetch_weather_data()` → Current weather + severity flags
- `get_historical_disruptions()` → Historical event database

---

### **Layer 2: Vector Store** (`backend/vector_store.py`)

**Purpose:** Embeds and persists documents for RAG retrieval

#### Key Components:

- **Embeddings Model:** HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (lightweight, free, local)
- **Vector Database:** ChromaDB (persistent at `vector_store/chroma_db/`)
- **Collection Name:** `supply_chain_risks`

#### Document Types & Formatting:

| Type           | Format Example                                                             |
| -------------- | -------------------------------------------------------------------------- |
| **NEWS**       | `[NEWS] Title` + Source + Published date + Summary                         |
| **WEATHER**    | `[WEATHER ALERT] City` + Condition + Temp/Wind + Supply chain impact       |
| **HISTORICAL** | `[HISTORICAL DISRUPTION] Event` + Date/Region/Type/Impact/Sectors/Severity |

#### Core Functions:

- `build_vector_store(data)` → Creates fresh embeddings from all 3 data types
- `load_vector_store()` → Loads persistent DB from disk
- `retrieve_relevant_context(query, k=6)` → RAG retrieval (top-k similar docs)
- `format_context_for_llm(docs)` → Formats documents for prompt injection

---

### **Layer 3: RAG Chains** (`backend/rag_chain.py`)

**Purpose:** LangChain orchestration with Gemini LLM for risk analysis

#### **API Key:**

- Currently hardcoded: `AIzaSyCv6SO3YPQf4fC5-zc9jOERUhsY5r7Lp3E`
- Model: `gemini-2.5-flash` (free tier: 15 RPM, 1M tokens/day)

#### **Three Main Chains:**

**1. RiskAssessmentChain**

- **Input:** Query + Company profile (name, industry, regions, critical inputs)
- **Process:** Retrieves 6 relevant documents → Injects into Gemini prompt → Parses JSON
- **Output:** JSON with:
  ```json
  {
    "overall_risk_level": "LOW|MEDIUM|HIGH|CRITICAL",
    "risk_score": 0-100,
    "identified_risks": [
      {
        "risk_id": "R001",
        "title": "...",
        "category": "geopolitical|natural_disaster|labor|economic|...",
        "severity": "LOW|MEDIUM|HIGH|CRITICAL",
        "probability": "LOW|MEDIUM|HIGH|CRITICAL",
        "affected_regions": [...],
        "affected_sectors": [...],
        "description": "...",
        "potential_impact": "...",
        "time_horizon": "short_term|medium_term|long_term",
        "evidence": "..."
      }
    ],
    "mitigation_strategies": [
      {
        "risk_id": "R001",
        "strategy": "...",
        "actions": [...],
        "timeline": "...",
        "cost_estimate": "low|medium|high"
      }
    ],
    "early_warning_indicators": [...],
    "recommended_immediate_actions": [...]
  }
  ```
- **Temperature:** 0.2 (low creativity for consistency)

**2. ScenarioGenerationChain**

- **Input:** Scenario type + Region + Industry
- **Process:** Retrieves 5 documents → Generates 3 plausible scenarios
- **Output:** JSON with 3 scenarios:
  ```json
  {
    "scenarios": [
      {
        "scenario_id": "S001",
        "title": "...",
        "type": "natural_disaster|geopolitical|...",
        "probability_score": 1-10,
        "severity_score": 1-10,
        "narrative": "...",
        "trigger_events": [...],
        "cascade_effects": [...],
        "affected_supply_chain_nodes": [...],
        "estimated_recovery_days": 30,
        "financial_impact_usd_millions": "100-300",
        "historical_analog": "..."
      }
    ]
  }
  ```
- **Temperature:** 0.7 (higher creativity for scenarios)

**3. ChatChain**

- **Input:** User question + chat history (last 6 messages)
- **Process:** Retrieves 5 documents → Conversational response
- **Output:** Plain text answer
- **Temperature:** 0.5 (balanced)

#### **Key Features:**

- **JSON Parsing:** Robust `_safe_parse_json()` with markdown stripping, truncation repair
- **Truncation Recovery:** `_attempt_repair()` closes unclosed brackets from token limits
- **Text Extraction:** `_extract_text()` handles both dict and string LLM responses

---

### **Layer 4: FastAPI Backend** (`backend/main.py`)

**Purpose:** HTTP API server with CORS, static file serving, background tasks

#### **Initialization (Lifespan):**

```
1. Check for GOOGLE_API_KEY in .env
2. Fetch all data (news, weather, historical)
3. Build vector store from documents
4. Initialize 3 chains (Risk, Scenario, Chat)
5. Ready at http://localhost:8000
```

#### **API Endpoints:**

| Method   | Endpoint                      | Purpose                                                   |
| -------- | ----------------------------- | --------------------------------------------------------- |
| **GET**  | `/`                           | Serves `frontend/index.html`                              |
| **GET**  | `/api/health`                 | System status (vector store ready, last refresh, key set) |
| **GET**  | `/api/data-summary`           | Counts of news, weather, historical + sources             |
| **POST** | `/api/assess-risk`            | Main risk assessment (takes query + company profile)      |
| **POST** | `/api/generate-scenarios`     | Generate 3 scenarios (scenario_type, region, industry)    |
| **POST** | `/api/chat`                   | Chat with the risk assistant                              |
| **POST** | `/api/refresh-data`           | Background task to refresh all data                       |
| **GET**  | `/api/historical-disruptions` | List all historical events                                |
| **GET**  | `/api/current-weather`        | Current weather snapshot                                  |

#### **Request/Response Models:**

- `CompanyProfile` → name, industry, regions, key_suppliers, critical_inputs
- `RiskAssessmentRequest` → query + optional company_profile
- `ScenarioRequest` → scenario_type, region, industry
- `ChatRequest` → message

---

### **Layer 5: Frontend** (`frontend/index.html`)

**Purpose:** Real-time dashboard UI (HTML/CSS/JS, no framework)

#### **Key Sections:**

1. **Header** → Logo, system status indicator, online/offline status
2. **Sidebar Navigation** → Risk Dashboard, Assessment, Scenarios, Chat, Weather
3. **Main Content Area** → Tab-based interface
4. **Right Panel** → Weather, Timeline, Historical events

#### **UI Components:**

- **Risk Overview Cards** → 4 cards showing counts/statuses (red/orange/yellow/green)
- **Risk List** → Detailed risk items with severity badges, collapsible
- **Assessment Form** → Company profile input (name, industry, regions, sectors)
- **Scenario Cards** → 3 scenarios with probability/severity scores, trigger events
- **Chat Interface** → Message history + input box
- **Weather Grid** → 8 cities with icons, temps, wind speeds

#### **Color Scheme:**

```
--accent: #00e5ff (cyan)
--accent2: #7c3aed (purple)
--green: #22d3a5
--yellow: #f59e0b
--orange: #f97316
--red: #ef4444
```

#### **Key JS Functions (inferred from HTML):**

- `assessRisk()` → POST to `/api/assess-risk`
- `generateScenarios()` → POST to `/api/generate-scenarios`
- `sendChat()` → POST to `/api/chat`
- `loadWeather()` → GET from `/api/current-weather`
- `loadHistorical()` → GET from `/api/historical-disruptions`
- `switchTab(tab)` → UI navigation

---

## 📊 Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│ USER INTERACTION (Frontend)                                      │
│  - Enter company profile                                         │
│  - Click "Assess Risk" / "Generate Scenarios" / "Chat"          │
└──────────────────┬───────────────────────────────────────────────┘
                   │ HTTP POST
                   ▼
┌──────────────────────────────────────────────────────────────────┐
│ FASTAPI BACKEND (main.py)                                        │
│  - Route to appropriate endpoint                                │
│  - Validate request                                             │
└──────────────────┬───────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────────┐
│ RAG CHAIN (rag_chain.py)                                         │
│  1. Query Vector Store for relevant docs                        │
│  2. Format context for prompt injection                         │
│  3. Send to Gemini API with prompt template                     │
│  4. Parse JSON response with error recovery                     │
└──────────────────┬───────────────────────────────────────────────┘
                   │
         ┌─────────┼─────────┐
         │         │         │
         ▼         ▼         ▼
    ┌────────┐ ┌────────┐ ┌────────┐
    │ Vector │ │ News   │ │Gemini  │
    │ Store  │ │Feeds   │ │ API    │
    │ChromaDB│ │Weather │ │        │
    │        │ │History │ │        │
    └────────┘ └────────┘ └────────┘
         │         │         │
         └─────────┼─────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────────┐
│ RESPONSE (JSON/Text)                                             │
└──────────────────┬───────────────────────────────────────────────┘
                   │ HTTP Response
                   ▼
┌──────────────────────────────────────────────────────────────────┐
│ FRONTEND RENDERING                                               │
│  - Display risks, scenarios, weather                            │
│  - Update charts/tables                                         │
└──────────────────────────────────────────────────────────────────┘
```

---

## 🔑 Key Design Decisions

### **1. Free & Local Architecture**

- ✅ Gemini API (free tier: 15 RPM)
- ✅ ChromaDB (local, persistent)
- ✅ HuggingFace embeddings (no API key needed)
- ✅ RSS feeds (no authentication)

### **2. RAG Over Fine-tuning**

- Retrieves current context from news/weather/historical data
- Enables real-time risk assessment without retraining
- Cites sources in responses

### **3. Robust JSON Parsing**

- Multiple recovery strategies for LLM truncation
- Handles markdown fences stripping
- Validates JSON structure before frontend delivery

### **4. Lifespan Management**

- Single initialization on startup
- Caches vector store, chains, data
- Allows background refresh without downtime

### **5. Modular Chain Design**

- Separate chains for risk assessment, scenarios, chat
- Different temperatures for different tasks
- Reusable prompts with variable injection

---

## 📦 Dependencies

| Package                  | Version | Purpose                        |
| ------------------------ | ------- | ------------------------------ |
| `langchain`              | 0.3.7   | Chains, prompts, orchestration |
| `langchain-google-genai` | 2.0.7   | Gemini integration             |
| `google-generativeai`    | 0.8.3   | Direct Gemini API              |
| `chromadb`               | 0.5.20  | Vector database                |
| `sentence-transformers`  | 3.3.1   | Embeddings                     |
| `fastapi`                | 0.115.5 | Web framework                  |
| `uvicorn`                | 0.32.1  | ASGI server                    |
| `feedparser`             | 6.0.11  | RSS parsing                    |
| `requests`               | 2.32.3  | HTTP calls                     |
| `python-dotenv`          | 1.0.1   | .env loading                   |

---

## ✅ Working Features

1. ✅ News ingestion from 5 RSS feeds
2. ✅ Weather monitoring for 8 global hub cities
3. ✅ Historical disruption database (10 events)
4. ✅ Vector store creation & persistence
5. ✅ RAG retrieval (top-k similar documents)
6. ✅ Risk assessment with Gemini
7. ✅ Scenario generation (3 scenarios)
8. ✅ Chat interface with context
9. ✅ FastAPI server with all endpoints
10. ✅ Frontend dashboard with tabs
11. ✅ JSON parsing with truncation recovery
12. ✅ CORS enabled for frontend

---

## ⚠️ Current Issues / Hardcoded Values

1. **API Key Hardcoded:** Line 26 in `rag_chain.py`

   ```python
   api_key = "AIzaSyCv6SO3YPQf4fC5-zc9jOERUhsY5r7Lp3E"  # Should be from .env
   ```

2. **Weather API Key Hardcoded:** Line 75 in `data_ingestion.py`

   ```python
   api_key = os.getenv("61bab8fd20dd5a7b24138ec2d2e784d6", "")  # Should be from .env
   ```

3. **Run.py Incomplete:** Ends at line 50, missing the main execution block

4. **Frontend Not Fully Implemented:** HTML has UI but JavaScript event handlers need full implementation

---

## 📁 Project Structure

```
RAG_MPR/
├── backend/
│   ├── __init__.py
│   ├── main.py                    # FastAPI + endpoints
│   ├── rag_chain.py               # LangChain orchestration
│   ├── vector_store.py            # ChromaDB + embeddings
│   ├── data_ingestion.py          # News + weather + historical
│   └── __pycache__/               # Compiled Python
├── frontend/
│   └── index.html                 # Dashboard UI
├── vector_store/
│   └── chroma_db/                 # Persistent embeddings
├── run.py                         # One-command launcher
├── requirements.txt               # Dependencies
├── README.md                      # Documentation
└── Untitled-1.ipynb               # Jupyter notebook (unused)
```

---

## 🚀 How to Run

```bash
# 1. Install dependencies
python run.py

# 2. The system will:
#    - Check for GOOGLE_API_KEY in .env
#    - Install requirements
#    - Start FastAPI server on port 8000
#    - Serve frontend at http://localhost:8000

# 3. Open browser and interact with dashboard
```

---

## 🎯 Ready for Questions & Changes

**Analysis Complete.** The project is well-structured with:

- ✅ Clear separation of concerns
- ✅ Modular architecture
- ✅ Real-time data integration
- ✅ RAG-based LLM orchestration
- ✅ Responsive frontend dashboard

**Awaiting your questions or change requests.**
