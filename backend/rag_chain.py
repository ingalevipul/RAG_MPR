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
    input_variables=["context", "company_profile", "query"],
    template="""You are a Supply Chain Risk Analyst. Respond with ONLY a JSON object — no markdown fences, no explanation, no text before or after the JSON.

Company: {company_profile}

Context (news / weather / historical data):
{context}

Query: {query}

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
      "evidence": "One sentence citing context."
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


# ── Chain classes ──────────────────────────────────────────────────────────────

class RiskAssessmentChain:
    def __init__(self, vectordb):
        self.vectordb = vectordb
        self.llm      = get_llm(temperature=0.2)
        self.chain    = LLMChain(llm=self.llm, prompt=RISK_ASSESSMENT_PROMPT)

    def run(self, query: str, company_profile: Optional[Dict] = None, k: int = 6) -> Dict[str, Any]:
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

        # ← invoke() replaces deprecated run()
        result_dict = self.chain.invoke({
            "context":         context,
            "company_profile": json.dumps(company_profile),
            "query":           query,
        })
        raw = _extract_text(result_dict)

        result = _safe_parse_json(raw)
        result["retrieved_sources"] = [
            {"type": d.metadata.get("type"), "preview": d.page_content[:100], "metadata": d.metadata}
            for d in docs
        ]
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

        result_dict = self.chain.invoke({
            "context":       context,
            "scenario_type": scenario_type,
            "region":        region,
            "industry":      industry,
        })
        raw = _extract_text(result_dict)
        return _safe_parse_json(raw)


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

        result_dict = self.chain.invoke({
            "context":      context,
            "chat_history": history_str,
            "question":     question,
        })
        answer = _extract_text(result_dict)

        self.history.append(f"User: {question}")
        self.history.append(f"Assistant: {answer}")
        return answer