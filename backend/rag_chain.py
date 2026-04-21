"""
rag_chain.py - LangChain RAG pipeline using Google Gemini (free tier)
Fixed: replaced deprecated chain.run() with chain.invoke()
Fixed: increased max_output_tokens to 8192, tighter prompts, JSON repair
"""

import os
import sys
import json
import re
from typing import List, Dict, Any, Optional

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_THIS_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(_ROOT_DIR, ".env"))

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


def get_llm(temperature: float = 0.7):
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not set. Get a free key at https://aistudio.google.com/apikey")
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=temperature,
        max_output_tokens=8192,
    )


# ── Prompts ────────────────────────────────────────────────────────────────────

RISK_ASSESSMENT_PROMPT = PromptTemplate(
    input_variables=["context", "realtime_data", "company_profile", "query"],
    template="""You are a Supply Chain Risk Analyst. Respond with ONLY a JSON object — no markdown fences, no explanation, no text before or after the JSON.

Company: {company_profile}

HISTORICAL & VECTOR-DB CONTEXT (previously indexed data):
{context}

REAL-TIME DATA (fetched just now — news, weather, live alerts):
{realtime_data}

Query: {query}

IMPORTANT: Give higher weight to the REAL-TIME DATA when assessing current risks, but also use historical context for patterns and analogies. Clearly cite whether evidence comes from real-time or historical data.

Return this exact JSON structure (keep all string values concise, max 2 sentences each):

{{
  "overall_risk_level": "LOW or MEDIUM or HIGH or CRITICAL",
  "risk_score": 75,
  "executive_summary": "Two sentence summary here.",
  "identified_risks": [
    {{
      "risk_id": "R001",
      "title": "Short title",
      "category": "geopolitical",
      "severity": "HIGH",
      "probability": "HIGH",
      "affected_regions": ["Asia Pacific"],
      "affected_sectors": ["Electronics"],
      "description": "One concise sentence.",
      "potential_impact": "One concise sentence.",
      "time_horizon": "short_term",
      "evidence": "One sentence citing context.",
      "data_source": "realtime or historical or both"
    }}
  ],
  "mitigation_strategies": [
    {{
      "risk_id": "R001",
      "strategy": "Strategy title",
      "actions": ["Action 1", "Action 2", "Action 3"],
      "timeline": "3-6 months",
      "cost_estimate": "medium"
    }}
  ],
  "early_warning_indicators": ["Indicator 1", "Indicator 2", "Indicator 3"],
  "recommended_immediate_actions": ["Action 1", "Action 2", "Action 3"]
}}

Rules:
- Include 3 to 4 risks maximum.
- Keep every string value short (1-2 sentences max).
- severity and probability must be exactly one of: LOW, MEDIUM, HIGH, CRITICAL.
- overall_risk_level must be exactly one of: LOW, MEDIUM, HIGH, CRITICAL.
- data_source must be exactly one of: realtime, historical, both.
- Output ONLY the JSON object. No markdown. No explanation.
"""
)

SCENARIO_GENERATION_PROMPT = PromptTemplate(
    input_variables=["context", "scenario_type", "region", "industry"],
    template="""You are a Supply Chain Risk Scenario Planner. Respond with ONLY a JSON object — no markdown, no explanation.

Context:
{context}

Parameters: type={scenario_type}, region={region}, industry={industry}

Return exactly this JSON (keep narratives to 2 sentences max):

{{
  "scenarios": [
    {{
      "scenario_id": "S001",
      "title": "Short scenario title",
      "type": "{scenario_type}",
      "probability_score": 7,
      "severity_score": 8,
      "narrative": "Two sentence description.",
      "trigger_events": ["Trigger 1", "Trigger 2"],
      "cascade_effects": ["Effect 1", "Effect 2", "Effect 3"],
      "affected_supply_chain_nodes": ["supplier", "manufacturing", "logistics"],
      "estimated_recovery_days": 60,
      "financial_impact_usd_millions": "100-300",
      "historical_analog": "Similar past event name"
    }},
    {{
      "scenario_id": "S002",
      "title": "Short scenario title",
      "type": "{scenario_type}",
      "probability_score": 5,
      "severity_score": 6,
      "narrative": "Two sentence description.",
      "trigger_events": ["Trigger 1", "Trigger 2"],
      "cascade_effects": ["Effect 1", "Effect 2", "Effect 3"],
      "affected_supply_chain_nodes": ["supplier", "logistics", "retail"],
      "estimated_recovery_days": 30,
      "financial_impact_usd_millions": "50-150",
      "historical_analog": "Similar past event name"
    }},
    {{
      "scenario_id": "S003",
      "title": "Short scenario title",
      "type": "{scenario_type}",
      "probability_score": 3,
      "severity_score": 9,
      "narrative": "Two sentence description.",
      "trigger_events": ["Trigger 1", "Trigger 2"],
      "cascade_effects": ["Effect 1", "Effect 2", "Effect 3"],
      "affected_supply_chain_nodes": ["manufacturing", "distribution"],
      "estimated_recovery_days": 180,
      "financial_impact_usd_millions": "500-1000",
      "historical_analog": "Similar past event name"
    }}
  ]
}}

Output ONLY the JSON object. No markdown. No explanation.
"""
)

CHAT_PROMPT = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template="""You are a Supply Chain Risk Intelligence Assistant.

Context from knowledge base:
{context}

Chat history:
{chat_history}

Question: {question}

Answer concisely and helpfully. Focus on actionable supply chain risk insights.
"""
)


# ── JSON parser ────────────────────────────────────────────────────────────────

def _safe_parse_json(text: str) -> Dict:
    """Robustly extract and parse JSON from LLM output, with truncation repair."""
    # Strip markdown fences
    text = re.sub(r"```(?:json)?", "", text).strip()

    # Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Extract outermost { ... } block, trying from the last } backwards
    start = text.find("{")
    if start != -1:
        for end in range(len(text) - 1, start, -1):
            if text[end] == "}":
                try:
                    return json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    continue

    # Attempt repair of truncated JSON
    fragment = text[start:] if start != -1 else text
    repaired = _attempt_repair(fragment)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    return {"error": "Failed to parse LLM response", "raw_output": text[:300]}


def _attempt_repair(text: str) -> str:
    """Close unclosed JSON brackets caused by token truncation."""
    stack = []
    in_string = False
    escape = False

    for ch in text:
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack and stack[-1] == ch:
            stack.pop()

    return text.rstrip().rstrip(",") + "".join(reversed(stack))


# ── Helper: extract text from invoke() result ──────────────────────────────────

def _extract_text(result: Any) -> str:
    """
    chain.invoke() returns a dict like {"text": "..."}
    Handle both dict and plain string responses.
    """
    if isinstance(result, dict):
        # LLMChain.invoke returns {"text": "...", ...}
        return result.get("text", result.get("output", str(result)))
    return str(result)


# ── Format real-time data for LLM prompt ───────────────────────────────────────

def _format_realtime_data(data: Dict) -> str:
    """Format freshly-fetched real-time data into a readable string for the LLM."""
    sections = []
    timestamp = data.get("fetch_timestamp", "unknown")
    sections.append(f"[Data fetched at: {timestamp}]")

    # News articles
    news = data.get("news", [])
    if news:
        sections.append(f"\n── LIVE NEWS ({len(news)} articles) ──")
        for i, article in enumerate(news[:10], 1):  # cap at 10 to save tokens
            sections.append(
                f"  {i}. [{article.get('source', 'unknown')}] {article.get('title', 'No title')}\n"
                f"     Published: {article.get('published', 'N/A')}\n"
                f"     Summary: {article.get('summary', 'N/A')[:200]}"
            )
    else:
        sections.append("\n── LIVE NEWS ── No relevant articles found at this time.")

    # Weather data
    weather = data.get("weather", [])
    if weather:
        sections.append(f"\n── LIVE WEATHER ({len(weather)} cities) ──")
        for w in weather:
            severity = "⚠ SEVERE" if w.get("is_severe") else "✓ Normal"
            sections.append(
                f"  • {w.get('city', '?')}: {w.get('description', 'N/A')} | "
                f"{w.get('temp_c', '?')}°C | Wind: {w.get('wind_kmh', '?')} km/h | {severity}"
            )
    else:
        sections.append("\n── LIVE WEATHER ── No weather data available.")

    return "\n".join(sections)


# ── Fallback builders (used when LLM quota is exhausted) ───────────────────────

_SEVERITY_MAP = {"thunderstorm": "HIGH", "rain": "MEDIUM", "snow": "MEDIUM",
                 "extreme": "CRITICAL", "tornado": "CRITICAL", "fog": "LOW"}


def _build_fallback_assessment(docs: List, realtime_data: Optional[Dict],
                               company_profile: Dict, query: str) -> Dict[str, Any]:
    """Build a structured risk assessment directly from raw data (no LLM)."""
    from datetime import datetime
    risks = []
    risk_id = 1

    # ── Risks from real-time weather ──
    if realtime_data:
        for w in realtime_data.get("weather", []):
            if w.get("is_severe"):
                sev = "HIGH"
                desc_lower = w.get("description", "").lower()
                for kw, s in _SEVERITY_MAP.items():
                    if kw in desc_lower:
                        sev = s
                        break
                risks.append({
                    "risk_id": f"R{risk_id:03d}",
                    "title": f"Severe Weather in {w['city']}",
                    "category": "weather",
                    "severity": sev,
                    "probability": "HIGH",
                    "affected_regions": [w["city"]],
                    "affected_sectors": ["shipping", "logistics", "manufacturing"],
                    "description": f"{w['description'].title()} — {w['temp_c']}°C, wind {w['wind_kmh']} km/h.",
                    "potential_impact": f"Port and logistics operations in {w['city']} may face delays.",
                    "time_horizon": "immediate",
                    "evidence": f"Live weather data fetched at {realtime_data.get('fetch_timestamp', 'N/A')}.",
                    "data_source": "realtime",
                })
                risk_id += 1

        # ── Risks from real-time news ──
        for article in realtime_data.get("news", [])[:4]:
            risks.append({
                "risk_id": f"R{risk_id:03d}",
                "title": article.get("title", "Unnamed Risk")[:80],
                "category": "geopolitical",
                "severity": "MEDIUM",
                "probability": "MEDIUM",
                "affected_regions": company_profile.get("regions", ["Global"]),
                "affected_sectors": [company_profile.get("industry", "Manufacturing")],
                "description": (article.get("summary", "") or "")[:200],
                "potential_impact": "Could affect supply chain operations. Monitor closely.",
                "time_horizon": "short_term",
                "evidence": f"Source: {article.get('source', 'news')} — {article.get('published', 'recent')}.",
                "data_source": "realtime",
            })
            risk_id += 1
            if risk_id > 4:
                break

    # ── Risks from vector-store historical docs (if we need more) ──
    if risk_id <= 4:
        for doc in docs:
            if risk_id > 4:
                break
            meta = doc.metadata
            if meta.get("type") == "historical":
                risks.append({
                    "risk_id": f"R{risk_id:03d}",
                    "title": meta.get("event", "Historical Disruption Pattern"),
                    "category": meta.get("type", "historical"),
                    "severity": (meta.get("severity", "medium")).upper(),
                    "probability": "MEDIUM",
                    "affected_regions": [meta.get("region", "Global")],
                    "affected_sectors": [company_profile.get("industry", "Manufacturing")],
                    "description": doc.page_content[:200],
                    "potential_impact": "Historical pattern suggests potential recurrence.",
                    "time_horizon": "medium_term",
                    "evidence": f"Historical record from {meta.get('date', 'N/A')}.",
                    "data_source": "historical",
                })
                risk_id += 1

    # ── Calculate overall risk ──
    sev_scores = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
    if risks:
        avg = sum(sev_scores.get(r["severity"], 2) for r in risks) / len(risks)
        if avg >= 3.5:
            overall, score = "CRITICAL", 90
        elif avg >= 2.5:
            overall, score = "HIGH", 75
        elif avg >= 1.5:
            overall, score = "MEDIUM", 50
        else:
            overall, score = "LOW", 25
    else:
        overall, score = "LOW", 20

    severe_cities = []
    if realtime_data:
        severe_cities = [w["city"] for w in realtime_data.get("weather", []) if w.get("is_severe")]

    return {
        "overall_risk_level": overall,
        "risk_score": score,
        "executive_summary": (
            f"⚠ AI model quota exceeded — this report was auto-generated from "
            f"{len(realtime_data.get('news', []) if realtime_data else [])} live news articles, "
            f"{len(realtime_data.get('weather', []) if realtime_data else [])} weather readings, "
            f"and {len(docs)} historical records. "
            f"{'Severe weather detected in: ' + ', '.join(severe_cities) + '.' if severe_cities else 'No severe weather alerts.'}"
        ),
        "identified_risks": risks[:4],
        "mitigation_strategies": [
            {
                "risk_id": r["risk_id"],
                "strategy": f"Monitor and mitigate: {r['title'][:40]}",
                "actions": [
                    "Set up alerts for this risk category",
                    "Contact affected suppliers for status updates",
                    "Prepare contingency logistics routes",
                ],
                "timeline": "1-3 months",
                "cost_estimate": "low",
            }
            for r in risks[:4]
        ],
        "early_warning_indicators": [
            "Increasing frequency of severe weather events in key port cities",
            "Rising geopolitical tensions in supply chain regions",
            "News volume spike for supply chain disruption keywords",
        ],
        "recommended_immediate_actions": [
            "Review current inventory levels for critical inputs",
            "Contact key suppliers in affected regions for status updates",
            "Activate backup logistics routes if severe weather persists",
        ],
    }


def _build_fallback_scenarios(docs: List, scenario_type: str,
                              region: str, industry: str) -> Dict[str, Any]:
    """Build fallback scenarios from vector-store docs when LLM is unavailable."""
    scenarios = []
    for i, doc in enumerate(docs[:3], 1):
        meta = doc.metadata
        scenarios.append({
            "scenario_id": f"S{i:03d}",
            "title": f"{scenario_type.replace('_', ' ').title()} Risk — {region}",
            "type": scenario_type,
            "probability_score": max(3, 7 - i),
            "severity_score": max(4, 8 - i),
            "narrative": doc.page_content[:250],
            "trigger_events": [
                f"{scenario_type.replace('_', ' ').title()} event in {region}",
                f"Supply chain disruption in {industry}",
            ],
            "cascade_effects": [
                "Supplier delays", "Shipping route disruptions", "Price increases"
            ],
            "affected_supply_chain_nodes": ["supplier", "manufacturing", "logistics"],
            "estimated_recovery_days": 60 + (i * 30),
            "financial_impact_usd_millions": f"{50 * i}-{150 * i}",
            "historical_analog": meta.get("event", "Similar past events"),
        })
    return {
        "scenarios": scenarios,
        "fallback": True,
        "note": "AI model quota exceeded — scenarios generated from historical data patterns.",
    }


def _build_fallback_chat(docs: List, question: str) -> str:
    """Build a fallback chat response from retrieved documents when LLM is unavailable."""
    lines = [
        "⚠ *AI model is temporarily unavailable (quota exceeded).* "
        "Here's what I found from the knowledge base:\n"
    ]
    if not docs:
        lines.append("No relevant documents found for your query.")
    else:
        for i, doc in enumerate(docs[:4], 1):
            meta = doc.metadata
            dtype = meta.get("type", "info").upper()
            lines.append(f"**{i}. [{dtype}]** {doc.page_content[:200]}...\n")
    lines.append("\n_Retry in ~60s when the API quota resets for a full AI-powered answer._")
    return "\n".join(lines)


# ── Chain classes ──────────────────────────────────────────────────────────────

class RiskAssessmentChain:
    def __init__(self, vectordb):
        self.vectordb = vectordb
        self.llm      = get_llm(temperature=0.2)
        self.chain    = LLMChain(llm=self.llm, prompt=RISK_ASSESSMENT_PROMPT)

    def run(self, query: str, company_profile: Optional[Dict] = None,
            realtime_data: Optional[Dict] = None, k: int = 6) -> Dict[str, Any]:
        from backend.vector_store import retrieve_relevant_context, format_context_for_llm

        if not company_profile:
            company_profile = {
                "name": "Generic Manufacturing Company",
                "industry": "Manufacturing",
                "regions": ["Asia Pacific", "Europe"],
                "critical_inputs": ["semiconductors", "raw materials"],
            }

        docs    = retrieve_relevant_context(query, self.vectordb, k=k)
        context = format_context_for_llm(docs)

        # Format real-time data for the prompt
        realtime_str = _format_realtime_data(realtime_data) if realtime_data else "No real-time data available."

        try:
            result_dict = self.chain.invoke({
                "context":         context,
                "realtime_data":   realtime_str,
                "company_profile": json.dumps(company_profile),
                "query":           query,
            })
            raw = _extract_text(result_dict)
            result = _safe_parse_json(raw)
        except Exception as e:
            error_msg = str(e).lower()
            print(f"[WARN] LLM call failed: {e}")
            # On quota / rate-limit / any LLM error → build fallback from raw data
            result = _build_fallback_assessment(docs, realtime_data, company_profile, query)
            result["fallback"] = True
            result["llm_error"] = str(e)[:200]

        result["retrieved_sources"] = [
            {"type": d.metadata.get("type"), "preview": d.page_content[:100], "metadata": d.metadata}
            for d in docs
        ]
        # Attach real-time data summary to the response
        if realtime_data:
            result["realtime_summary"] = {
                "news_count":    len(realtime_data.get("news", [])),
                "weather_count": len(realtime_data.get("weather", [])),
                "severe_weather": [w["city"] for w in realtime_data.get("weather", []) if w.get("is_severe")],
                "fetch_timestamp": realtime_data.get("fetch_timestamp", ""),
            }
        return result


class ScenarioGenerationChain:
    def __init__(self, vectordb):
        self.vectordb = vectordb
        self.llm      = get_llm(temperature=0.7)
        self.chain    = LLMChain(llm=self.llm, prompt=SCENARIO_GENERATION_PROMPT)

    def run(self, scenario_type: str = "natural_disaster", region: str = "Asia Pacific", industry: str = "Electronics") -> Dict[str, Any]:
        from backend.vector_store import retrieve_relevant_context, format_context_for_llm

        query   = f"{scenario_type} supply chain disruption {region} {industry}"
        docs    = retrieve_relevant_context(query, self.vectordb, k=5)
        context = format_context_for_llm(docs)

        try:
            result_dict = self.chain.invoke({
                "context":       context,
                "scenario_type": scenario_type,
                "region":        region,
                "industry":      industry,
            })
            raw = _extract_text(result_dict)
            return _safe_parse_json(raw)
        except Exception as e:
            print(f"[WARN] Scenario LLM call failed: {e}")
            return _build_fallback_scenarios(docs, scenario_type, region, industry)


class ChatChain:
    def __init__(self, vectordb):
        self.vectordb = vectordb
        self.llm      = get_llm(temperature=0.5)
        self.chain    = LLMChain(llm=self.llm, prompt=CHAT_PROMPT)
        self.history: List[str] = []

    def run(self, question: str) -> str:
        from backend.vector_store import retrieve_relevant_context, format_context_for_llm

        docs        = retrieve_relevant_context(question, self.vectordb, k=5)
        context     = format_context_for_llm(docs)
        history_str = "\n".join(self.history[-6:]) if self.history else "None"

        try:
            result_dict = self.chain.invoke({
                "context":      context,
                "chat_history": history_str,
                "question":     question,
            })
            answer = _extract_text(result_dict)
        except Exception as e:
            print(f"[WARN] Chat LLM call failed: {e}")
            # Build a fallback chat response from retrieved docs
            answer = _build_fallback_chat(docs, question)

        self.history.append(f"User: {question}")
        self.history.append(f"Assistant: {answer}")
        return answer