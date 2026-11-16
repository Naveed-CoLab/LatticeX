"""Rank today's stocks with the saved model. The dashboard runs this itself after each close.

    python -m scripts.score_today

Writes data/live/rank.json and data/live/forecasts/<day>.csv. See src/models/production.py.
"""
from src.models.production import score

if __name__ == "__main__":
    score()
