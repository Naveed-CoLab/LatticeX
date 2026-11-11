"""Check that every API key in .env works. Run: python -m scripts.check_keys
Needs: pip install python-dotenv requests
"""
import os
import requests
from dotenv import load_dotenv

load_dotenv()


def check(name, fn):
    try:
        print(f"[OK]   {name}: {fn()}")
    except Exception as e:
        print(f"[FAIL] {name}: {e}")


def fred():
    key = os.getenv("FRED_API_KEY")
    r = requests.get(
        "https://api.stlouisfed.org/fred/series/observations",
        params={"series_id": "VIXCLS", "api_key": key, "file_type": "json",
                "sort_order": "desc", "limit": 1},
        timeout=20,
    )
    r.raise_for_status()
    obs = r.json()["observations"][0]
    return f"latest VIX {obs['value']} on {obs['date']}"


def hf():
    r = requests.get(
        "https://huggingface.co/api/whoami-v2",
        headers={"Authorization": f"Bearer {os.getenv('HF_TOKEN')}"},
        timeout=20,
    )
    r.raise_for_status()
    d = r.json()
    return f"user {d['name']}, token role {d['auth']['accessToken']['role']}"


def benzinga():
    r = requests.get(
        "https://api.benzinga.com/api/v2/news",
        params={"token": os.getenv("BENZINGA_API_KEY"), "tickers": "AAPL",
                "pageSize": 1, "displayOutput": "headline"},
        headers={"accept": "application/json"},
        timeout=20,
    )
    r.raise_for_status()
    items = r.json()
    return f"latest AAPL headline: {items[0]['title'][:70]}" if items else "no articles returned"


def sec():
    ua = os.getenv("SEC_USER_AGENT")
    r = requests.get(
        "https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json",
        headers={"User-Agent": ua}, timeout=30,
    )
    r.raise_for_status()
    return f"Apple companyfacts loaded ({len(r.json()['facts']['us-gaap'])} us-gaap fields)"


def tiingo():
    key = os.getenv("TIINGO_API_KEY")
    if not key:
        return "not set yet (skipped)"
    r = requests.get(
        "https://api.tiingo.com/tiingo/daily/AAPL/prices",
        params={"startDate": "2000-01-03", "endDate": "2000-01-05", "token": key},
        timeout=20,
    )
    r.raise_for_status()
    return f"AAPL 2000-01-03 adjClose {r.json()[0]['adjClose']:.4f}"


for name, fn in [("FRED", fred), ("Hugging Face", hf), ("Benzinga", benzinga),
                 ("SEC EDGAR", sec), ("Tiingo", tiingo)]:
    check(name, fn)
