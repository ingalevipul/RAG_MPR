"""
data_ingestion.py
Fetches real-time data from free sources:
  - RSS news feeds (Reuters, BBC)
  - OpenWeatherMap (free tier)
  - Historical disruption records (built-in dataset)
"""

import feedparser
import requests
import json
import os
from datetime import datetime
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# 1. NEWS FEEDS  (completely free, no API key)
# ─────────────────────────────────────────────
RSS_FEEDS = {
    "reuters_world":  "https://feeds.reuters.com/reuters/worldNews",
    "reuters_biz":    "https://feeds.reuters.com/reuters/businessNews",
    "bbc_world":      "https://feeds.bbci.co.uk/news/world/rss.xml",
    "reliefweb":      "https://reliefweb.int/disasters/rss.xml",   # disaster alerts
    "gdacs":          "https://www.gdacs.org/xml/rss.xml",          # global disaster alerts
}

RISK_KEYWORDS = [
    "supply chain", "disruption", "shortage", "flood", "earthquake",
    "hurricane", "typhoon", "war", "conflict", "sanction", "strike",
    "port", "shipping", "logistics", "factory", "semiconductor",
    "geopolitical", "trade war", "tariff", "blockade", "pandemic",
    "drought", "wildfire", "volcano", "tsunami", "recession",
]


def fetch_news_articles(max_per_feed: int = 10) -> List[Dict]:
    """Pull recent articles from RSS feeds, filter by supply-chain keywords."""
    articles = []
    for source, url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:max_per_feed]:
                title   = entry.get("title", "")
                summary = entry.get("summary", entry.get("description", ""))
                text    = (title + " " + summary).lower()

                if any(kw in text for kw in RISK_KEYWORDS):
                    articles.append({
                        "source":    source,
                        "title":     title,
                        "summary":   summary,
                        "link":      entry.get("link", ""),
                        "published": entry.get("published", str(datetime.now())),
                        "type":      "news",
                    })
        except Exception as e:
            print(f"[WARN] Feed {source} failed: {e}")
    return articles


# ─────────────────────────────────────────────
# 2. WEATHER ALERTS  (OpenWeatherMap free tier)
# ─────────────────────────────────────────────
MAJOR_SUPPLY_CHAIN_CITIES = [
    {"name": "Shanghai",    "lat": 31.2304, "lon": 121.4737},
    {"name": "Rotterdam",   "lat": 51.9244, "lon":   4.4777},
    {"name": "Singapore",   "lat":  1.3521, "lon": 103.8198},
    {"name": "Los Angeles", "lat": 34.0522, "lon": -118.2437},
    {"name": "Dubai",       "lat": 25.2048, "lon":  55.2708},
    {"name": "Navi Mumbai", "lat": 19.0760, "lon":  72.8777},
    {"name": "Mumbai",      "lat": 19.0760, "lon":  72.8777},
    {"name": "Surat",       "lat": 21.1702, "lon":  72.8311},
]

SEVERE_WEATHER_CODES = {
    range(200, 300): "Thunderstorm",
    range(300, 400): "Drizzle",
    range(500, 600): "Rain",
    range(600, 700): "Snow",
    range(700, 800): "Atmospheric hazard (fog/smoke/tornado)",
    range(900, 910): "Extreme weather",
}


def fetch_weather_data() -> List[Dict]:
    """Fetch current weather for major port/logistics cities."""
    api_key = os.getenv("WEATHER_API_KEY", "")
    results = []

    if not api_key:
        # Return mock data so the system works without a key
        return _mock_weather_data()

    for city in MAJOR_SUPPLY_CHAIN_CITIES:
        try:
            url = (
                f"https://api.openweathermap.org/data/2.5/weather"
                f"?lat={city['lat']}&lon={city['lon']}"
                f"&appid={api_key}&units=metric"
            )
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                weather_id   = data["weather"][0]["id"]
                description  = data["weather"][0]["description"]
                temp         = data["main"]["temp"]
                wind_speed   = data["wind"]["speed"]
                is_severe    = weather_id < 700 or weather_id >= 900

                results.append({
                    "city":        city["name"],
                    "weather_id":  weather_id,
                    "description": description,
                    "temp_c":      temp,
                    "wind_kmh":    round(wind_speed * 3.6, 1),
                    "is_severe":   is_severe,
                    "timestamp":   str(datetime.now()),
                    "type":        "weather",
                })
        except Exception as e:
            print(f"[WARN] Weather fetch failed for {city['name']}: {e}")

    return results


def _mock_weather_data() -> List[Dict]:
    """Fallback mock weather when no API key provided."""
    return [
        {
            "city": "Shanghai", "weather_id": 501,
            "description": "moderate rain", "temp_c": 18, "wind_kmh": 22,
            "is_severe": True, "timestamp": str(datetime.now()), "type": "weather",
        },
        {
            "city": "Rotterdam", "weather_id": 800,
            "description": "clear sky", "temp_c": 12, "wind_kmh": 15,
            "is_severe": False, "timestamp": str(datetime.now()), "type": "weather",
        },
        {
            "city": "Singapore", "weather_id": 200,
            "description": "thunderstorm with light rain", "temp_c": 30, "wind_kmh": 35,
            "is_severe": True, "timestamp": str(datetime.now()), "type": "weather",
        },
    ]


# ─────────────────────────────────────────────
# 3. HISTORICAL DISRUPTION RECORDS  (built-in)
# ─────────────────────────────────────────────
HISTORICAL_DISRUPTIONS = [
    {
        "event": "Suez Canal Blockage (Ever Given)",
        "date": "2021-03-23",
        "region": "Middle East / Egypt",
        "type": "logistics",
        "impact": "Blocked ~12% of global trade for 6 days. $9.6B/day in delayed goods.",
        "affected_sectors": ["shipping", "oil", "manufacturing", "retail"],
        "resolution_days": 6,
        "severity": "critical",
    },
    {
        "event": "COVID-19 Pandemic Supply Disruption",
        "date": "2020-03-01",
        "region": "Global",
        "type": "pandemic",
        "impact": "Factory shutdowns in China, Southeast Asia. PPE/semiconductor shortages.",
        "affected_sectors": ["electronics", "automotive", "pharma", "retail"],
        "resolution_days": 730,
        "severity": "critical",
    },
    {
        "event": "Fukushima Nuclear Disaster",
        "date": "2011-03-11",
        "region": "Japan",
        "type": "natural_disaster",
        "impact": "Auto/electronics supply chains disrupted. Toyota halted production globally.",
        "affected_sectors": ["automotive", "electronics"],
        "resolution_days": 180,
        "severity": "high",
    },
    {
        "event": "Taiwan Semiconductor Shortage",
        "date": "2021-01-01",
        "region": "Asia Pacific",
        "type": "geopolitical",
        "impact": "Global chip shortage affecting auto, consumer electronics, appliances.",
        "affected_sectors": ["automotive", "electronics", "appliances"],
        "resolution_days": 540,
        "severity": "high",
    },
    {
        "event": "Russia-Ukraine War Trade Impact",
        "date": "2022-02-24",
        "region": "Eastern Europe",
        "type": "geopolitical",
        "impact": "Wheat, sunflower oil, neon gas (semiconductor manufacturing) shortages.",
        "affected_sectors": ["food", "energy", "semiconductors", "fertilizers"],
        "resolution_days": 999,
        "severity": "critical",
    },
    {
        "event": "US-China Trade War Tariffs",
        "date": "2018-07-06",
        "region": "USA / China",
        "type": "geopolitical",
        "impact": "25% tariffs on $250B Chinese goods. Companies shifted manufacturing to Vietnam.",
        "affected_sectors": ["electronics", "manufacturing", "agriculture"],
        "resolution_days": 999,
        "severity": "high",
    },
    {
        "event": "Thailand Floods",
        "date": "2011-07-01",
        "region": "Southeast Asia",
        "type": "natural_disaster",
        "impact": "Hard drive production dropped 30%. Flooding of 7 major industrial estates.",
        "affected_sectors": ["electronics", "automotive"],
        "resolution_days": 180,
        "severity": "high",
    },
    {
        "event": "Los Angeles Port Strike",
        "date": "2014-11-01",
        "region": "North America",
        "type": "labor",
        "impact": "Slowdown at busiest US port. Retail import delays of 2-4 weeks.",
        "affected_sectors": ["retail", "manufacturing", "agriculture"],
        "resolution_days": 120,
        "severity": "medium",
    },
    {
        "event": "Yemen Houthi Red Sea Attacks",
        "date": "2023-11-19",
        "region": "Middle East",
        "type": "geopolitical",
        "impact": "Shipping companies rerouted around Cape of Good Hope adding 10-14 days.",
        "affected_sectors": ["shipping", "oil", "retail", "manufacturing"],
        "resolution_days": 365,
        "severity": "high",
    },
    {
        "event": "Panama Canal Drought Water Shortage",
        "date": "2023-09-01",
        "region": "Central America",
        "type": "natural_disaster",
        "impact": "Ships limited to lower capacity. Queues of 100+ vessels. Passage fees tripled.",
        "affected_sectors": ["shipping", "agriculture", "energy"],
        "resolution_days": 180,
        "severity": "medium",
    },
]


def get_historical_disruptions() -> List[Dict]:
    return [dict(d, type_label="historical_disruption") for d in HISTORICAL_DISRUPTIONS]


# ─────────────────────────────────────────────
# 4. AGGREGATE ALL DATA
# ─────────────────────────────────────────────
def fetch_all_data() -> Dict[str, List[Dict]]:
    print("[INFO] Fetching news articles...")
    news = fetch_news_articles()

    print("[INFO] Fetching weather data...")
    weather = fetch_weather_data()

    print("[INFO] Loading historical disruptions...")
    historical = get_historical_disruptions()

    print(f"[INFO] Data fetched — News: {len(news)}, Weather: {len(weather)}, Historical: {len(historical)}")
    return {
        "news":       news,
        "weather":    weather,
        "historical": historical,
    }


if __name__ == "__main__":
    data = fetch_all_data()
    print(json.dumps(data, indent=2, default=str))
