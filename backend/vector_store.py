"""
vector_store.py
Builds and queries a ChromaDB vector store using free HuggingFace embeddings.
No paid API needed for embeddings — uses sentence-transformers locally.
"""

import os
import sys
from typing import List, Dict

# Ensure project root is on path (needed when run directly)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_THIS_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# ── Config ──
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DIR  = os.path.join(_ROOT_DIR, "vector_store", "chroma_db")
COLLECTION  = "supply_chain_risks"


def get_embeddings():
    """Return HuggingFace embedding model (downloaded once, cached locally)."""
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def _news_to_doc(item: Dict) -> Document:
    content = (
        f"[NEWS] {item['title']}\n"
        f"Source: {item['source']}\n"
        f"Published: {item['published']}\n"
        f"Summary: {item['summary']}"
    )
    return Document(
        page_content=content,
        metadata={
            "type":   "news",
            "source": item["source"],
            "link":   item.get("link", ""),
            "date":   item.get("published", ""),
        },
    )


def _weather_to_doc(item: Dict) -> Document:
    severity = "SEVERE" if item["is_severe"] else "normal"
    content = (
        f"[WEATHER ALERT - {severity}] {item['city']}\n"
        f"Condition: {item['description']}\n"
        f"Temperature: {item['temp_c']}C, Wind: {item['wind_kmh']} km/h\n"
        f"Supply chain risk: {'High - extreme weather may disrupt port operations' if item['is_severe'] else 'Low'}\n"
        f"Timestamp: {item['timestamp']}"
    )
    return Document(
        page_content=content,
        metadata={
            "type":      "weather",
            "city":      item["city"],
            "is_severe": str(item["is_severe"]),
            "date":      item["timestamp"],
        },
    )


def _historical_to_doc(item: Dict) -> Document:
    content = (
        f"[HISTORICAL DISRUPTION] {item['event']}\n"
        f"Date: {item['date']}\n"
        f"Region: {item['region']}\n"
        f"Type: {item['type']}\n"
        f"Impact: {item['impact']}\n"
        f"Affected sectors: {', '.join(item['affected_sectors'])}\n"
        f"Severity: {item['severity']}\n"
        f"Resolution time: {item['resolution_days']} days"
    )
    return Document(
        page_content=content,
        metadata={
            "type":     "historical",
            "event":    item["event"],
            "region":   item["region"],
            "severity": item["severity"],
            "date":     item["date"],
        },
    )


def build_vector_store(data: Dict[str, List[Dict]]) -> Chroma:
    """Ingest all data sources into ChromaDB."""
    print("[INFO] Building vector store...")
    os.makedirs(CHROMA_DIR, exist_ok=True)

    docs: List[Document] = []
    for item in data.get("news", []):
        docs.append(_news_to_doc(item))
    for item in data.get("weather", []):
        docs.append(_weather_to_doc(item))
    for item in data.get("historical", []):
        docs.append(_historical_to_doc(item))

    if not docs:
        raise ValueError("No documents to index!")

    embeddings = get_embeddings()
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION,
    )
    vectordb.persist()
    print(f"[INFO] Vector store built with {len(docs)} documents.")
    return vectordb


def load_vector_store() -> Chroma:
    """Load existing ChromaDB from disk."""
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION,
    )


def retrieve_relevant_context(query: str, vectordb: Chroma, k: int = 8) -> List[Document]:
    retriever = vectordb.as_retriever(search_kwargs={"k": k})
    docs = retriever.get_relevant_documents(query)
    return docs


def format_context_for_llm(docs: List[Document]) -> str:
    sections = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        sections.append(
            f"--- Source {i} [{meta.get('type', 'unknown').upper()}] ---\n"
            f"{doc.page_content}"
        )
    return "\n\n".join(sections)
