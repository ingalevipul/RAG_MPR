"""
rag_chain.py
LangChain RAG pipeline using Google Gemini (free tier).
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
from langchain.schema import Document


def get_llm(temperature: float = 0.7):
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY not set. Get a free key at https://aistudio.google.com/apikey"
        )
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=temperature,
        max_output_tokens=2048,
    )


RISK_ASSESSMENT_PROMPT = PromptTemplate(
    input_variables=["context", "company_profile", "query"],
    template="""You are an expert Supply Chain Risk Analyst with deep knowledge of global logistics,
geopolitics, and natural disasters. Your job is to proactively identify and assess risks.

## Company Profile
{company_profile}

## Retrieved Real-Time & Historical Context
{context}

## User Query / Focus Area
{query}

## Instructions
Based on the context above, generate a comprehensive supply chain risk assessment.
Return your response as a valid JSON object with this exact structure:

{{
  "overall_risk_level": "LOW|MEDIUM|HIGH|CRITICAL",
  "risk_score": <integer 1-100>,
  "executive_summary": "<2-3 sentence summary>",
  "identified_risks": [
    {{
      "risk_id": "R001",
      "title": "<short risk title>",
      "category": "natural_disaster|geopolitical|logistics|economic|labor|cyber|pandemic",
      "severity": "LOW|MEDIUM|HIGH|CRITICAL",
      "probability": "LOW|MEDIUM|HIGH",
      "affected_regions": ["<region1>"],
      "affected_sectors": ["<sector1>"],
      "description": "<detailed description>",
      "potential_impact": "<business impact>",
      "time_horizon": "immediate|short_term|medium_term|long_term",
      "evidence": "<what from the context supports this risk>"
    }}
  ],
  "mitigation_strategies": [
    {{
      "risk_id": "R001",
      "strategy": "<strategy title>",
      "actions": ["<action1>", "<action2>", "<action3>"],
      "timeline": "<implementation timeline>",
      "cost_estimate": "low|medium|high"
    }}
  ],
  "early_warning_indicators": ["<indicator1>", "<indicator2>", "<indicator3>"],
  "recommended_immediate_actions": ["<action1>", "<action2>"]
}}

Return ONLY the JSON object, no markdown, no explanation.
"""
)

SCENARIO_GENERATION_PROMPT = PromptTemplate(
    input_variables=["context", "scenario_type", "region", "industry"],
    template="""You are a Supply Chain Risk Scenario Planner. Generate realistic risk scenarios
based on real-world patterns and current events.

## Context from Knowledge Base
{context}

## Scenario Parameters
- Type: {scenario_type}
- Region: {region}
- Industry: {industry}

Generate 3 plausible risk scenarios. Return as JSON:

{{
  "scenarios": [
    {{
      "scenario_id": "S001",
      "title": "<scenario title>",
      "type": "{scenario_type}",
      "probability_score": <1-10>,
      "severity_score": <1-10>,
      "narrative": "<2-3 sentence realistic scenario description>",
      "trigger_events": ["<event1>", "<event2>"],
      "cascade_effects": ["<effect1>", "<effect2>", "<effect3>"],
      "affected_supply_chain_nodes": ["supplier", "manufacturing", "logistics"],
      "estimated_recovery_days": <number>,
      "financial_impact_usd_millions": "<range e.g. 50-200>",
      "historical_analog": "<similar past event>"
    }}
  ]
}}

Return ONLY the JSON, no markdown.
"""
)

CHAT_PROMPT = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template="""You are a Supply Chain Risk Intelligence Assistant. Answer questions using the
retrieved context from news, weather, and historical disruption data.

## Retrieved Context
{context}

## Conversation History
{chat_history}

## Current Question
{question}

Provide a helpful, accurate, and concise answer. If the context doesn't contain enough
information, say so clearly but still provide general expert knowledge.
"""
)


def _safe_parse_json(text: str) -> Dict:
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
    return {"error": "Failed to parse LLM response", "raw_output": text[:500]}


class RiskAssessmentChain:
    def __init__(self, vectordb):
        self.vectordb = vectordb
        self.llm      = get_llm(temperature=0.3)
        self.chain    = LLMChain(llm=self.llm, prompt=RISK_ASSESSMENT_PROMPT)

    def run(self, query: str, company_profile: Optional[Dict] = None, k: int = 8) -> Dict[str, Any]:
        from backend.vector_store import retrieve_relevant_context, format_context_for_llm
        if not company_profile:
            company_profile = {
                "name": "Generic Manufacturing Company",
                "industry": "Manufacturing",
                "regions": ["Asia Pacific", "Europe", "North America"],
                "key_suppliers": ["China", "Vietnam", "Germany"],
                "critical_inputs": ["semiconductors", "raw materials", "logistics"],
            }
        docs    = retrieve_relevant_context(query, self.vectordb, k=k)
        context = format_context_for_llm(docs)
        raw     = self.chain.run(
            context=context,
            company_profile=json.dumps(company_profile, indent=2),
            query=query,
        )
        result = _safe_parse_json(raw)
        result["retrieved_sources"] = [
            {"type": d.metadata.get("type"), "preview": d.page_content[:100], "metadata": d.metadata}
            for d in docs
        ]
        return result


class ScenarioGenerationChain:
    def __init__(self, vectordb):
        self.vectordb = vectordb
        self.llm      = get_llm(temperature=0.8)
        self.chain    = LLMChain(llm=self.llm, prompt=SCENARIO_GENERATION_PROMPT)

    def run(self, scenario_type: str = "natural_disaster", region: str = "Asia Pacific", industry: str = "Electronics") -> Dict[str, Any]:
        from backend.vector_store import retrieve_relevant_context, format_context_for_llm
        query   = f"{scenario_type} risk {region} {industry} supply chain"
        docs    = retrieve_relevant_context(query, self.vectordb, k=6)
        context = format_context_for_llm(docs)
        raw     = self.chain.run(context=context, scenario_type=scenario_type, region=region, industry=industry)
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
        history_str = "\n".join(self.history[-6:]) if self.history else "No previous messages."
        answer      = self.chain.run(context=context, chat_history=history_str, question=question)
        self.history.append(f"User: {question}")
        self.history.append(f"Assistant: {answer}")
        return answer
