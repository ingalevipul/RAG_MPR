"""
main.py  —  FastAPI backend for Supply Chain Risk Assessment

HOW TO RUN (from inside the supply_chain_risk/ folder):
    python run.py                                          <- recommended
    uvicorn backend.main:app --reload --port 8000          <- manual
"""

import os
import sys
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from contextlib import asynccontextmanager

# Fix sys.path so `backend.*` imports work regardless of CWD or OS
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_THIS_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv(os.path.join(_ROOT_DIR, ".env"))

_FRONTEND_DIR = os.path.join(_ROOT_DIR, "frontend")
_VECTOR_DIR   = os.path.join(_ROOT_DIR, "vector_store", "chroma_db")

_vectordb       = None
_data_cache     = None
_last_refresh   = None
_risk_chain     = None
_scenario_chain = None
_chat_chain     = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[STARTUP] Initializing Supply Chain Risk System...")
    await _initialize_system()
    yield
    print("[SHUTDOWN] Cleaning up...")


async def _initialize_system():
    global _vectordb, _data_cache, _last_refresh, _risk_chain, _scenario_chain, _chat_chain

    from backend.data_ingestion import fetch_all_data
    from backend.vector_store import build_vector_store, load_vector_store
    from backend.rag_chain import RiskAssessmentChain, ScenarioGenerationChain, ChatChain

    os.makedirs(_VECTOR_DIR, exist_ok=True)

    try:
        _data_cache   = fetch_all_data()
        _vectordb     = build_vector_store(_data_cache)
        _last_refresh = datetime.now().isoformat()
        print("[STARTUP] Vector store built successfully.")
    except Exception as e:
        print(f"[WARN] Could not build fresh vector store: {e}")
        try:
            _vectordb = load_vector_store()
            print("[STARTUP] Loaded existing vector store from disk.")
        except Exception as e2:
            print(f"[ERROR] No vector store available: {e2}")
            _vectordb = None

    if _vectordb:
        _risk_chain     = RiskAssessmentChain(_vectordb)
        _scenario_chain = ScenarioGenerationChain(_vectordb)
        _chat_chain     = ChatChain(_vectordb)
        print("[STARTUP] All chains initialized. System ready.")
    else:
        print("[WARN] Chains NOT initialized - vector store unavailable.")


app = FastAPI(
    title="Supply Chain Risk Assessment API",
    description="Proactive supply chain risk assessment using RAG + Gemini",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.exists(_FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=_FRONTEND_DIR), name="static")


class CompanyProfile(BaseModel):
    name:            str       = "My Company"
    industry:        str       = "Manufacturing"
    regions:         List[str] = ["Asia Pacific"]
    key_suppliers:   List[str] = ["China"]
    critical_inputs: List[str] = ["semiconductors"]


class RiskAssessmentRequest(BaseModel):
    query:           str                       = "Assess current supply chain risks"
    company_profile: Optional[CompanyProfile]  = None


class ScenarioRequest(BaseModel):
    scenario_type: str = "natural_disaster"
    region:        str = "Asia Pacific"
    industry:      str = "Electronics"


class ChatRequest(BaseModel):
    message: str


@app.get("/", response_class=HTMLResponse)
async def root():
    index = os.path.join(_FRONTEND_DIR, "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    return HTMLResponse("<h1>Supply Chain Risk API</h1><p>Visit <a href='/docs'>/docs</a></p>")


@app.get("/api/health")
async def health():
    return {
        "status":       "ok",
        "vector_store": "ready" if _vectordb else "unavailable",
        "last_refresh": _last_refresh,
        "api_key_set":  bool(os.getenv("GOOGLE_API_KEY")),
    }


@app.get("/api/data-summary")
async def data_summary():
    if not _data_cache:
        raise HTTPException(503, "Data not yet loaded.")
    return {
        "news_articles":     len(_data_cache.get("news", [])),
        "weather_readings":  len(_data_cache.get("weather", [])),
        "historical_events": len(_data_cache.get("historical", [])),
        "last_refresh":      _last_refresh,
        "news_sources":      list({n["source"] for n in _data_cache.get("news", [])}),
        "weather_cities":    [w["city"] for w in _data_cache.get("weather", [])],
        "severe_weather":    [w["city"] for w in _data_cache.get("weather", []) if w.get("is_severe")],
    }


@app.post("/api/assess-risk")
async def assess_risk(req: RiskAssessmentRequest):
    if not _risk_chain:
        raise HTTPException(503, "Risk chain not initialized. Check GOOGLE_API_KEY in .env")
    try:
        from backend.data_ingestion import fetch_news_articles, fetch_weather_data

        # ── Fetch real-time data at assessment time ──
        print("[ASSESS] Fetching real-time news articles...")
        realtime_news = fetch_news_articles(max_per_feed=5)
        print(f"[ASSESS] Got {len(realtime_news)} real-time news articles")

        print("[ASSESS] Fetching real-time weather data...")
        realtime_weather = fetch_weather_data()
        print(f"[ASSESS] Got {len(realtime_weather)} weather readings")

        realtime_data = {
            "news":            realtime_news,
            "weather":         realtime_weather,
            "fetch_timestamp": datetime.now().isoformat(),
        }

        profile_dict = req.company_profile.dict() if req.company_profile else None
        result = _risk_chain.run(
            query=req.query,
            company_profile=profile_dict,
            realtime_data=realtime_data,
        )
        return {
            "success":   True,
            "data":      result,
            "timestamp": datetime.now().isoformat(),
            "realtime_fetch": {
                "news_articles":    len(realtime_news),
                "weather_readings": len(realtime_weather),
                "severe_weather":   [w["city"] for w in realtime_weather if w.get("is_severe")],
            },
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/generate-scenarios")
async def generate_scenarios(req: ScenarioRequest):
    if not _scenario_chain:
        raise HTTPException(503, "Scenario chain not initialized.")
    try:
        result = _scenario_chain.run(
            scenario_type=req.scenario_type,
            region=req.region,
            industry=req.industry,
        )
        return {"success": True, "data": result, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/chat")
async def chat(req: ChatRequest):
    if not _chat_chain:
        raise HTTPException(503, "Chat chain not initialized.")
    try:
        answer = _chat_chain.run(req.message)
        return {"success": True, "answer": answer, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/refresh-data")
async def refresh_data(background_tasks: BackgroundTasks):
    background_tasks.add_task(_initialize_system)
    return {"message": "Data refresh started - check /api/health in ~30s"}


@app.get("/api/historical-disruptions")
async def get_historical():
    from backend.data_ingestion import get_historical_disruptions
    return {"data": get_historical_disruptions()}


@app.get("/api/current-weather")
async def get_weather():
    from backend.data_ingestion import fetch_weather_data
    return {"data": fetch_weather_data()}
