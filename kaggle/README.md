# kaggle/

Code that runs **inside** Kaggle notebooks. Most jobs don't need anything here: `scripts/kaggle_cloud.py` zips the repo into the notebook and runs the normal `python -m ...` commands. See [`../docs/kaggle-workflow.md`](../docs/kaggle-workflow.md).

| Path | Purpose |
|---|---|
| `finbert/finbert_score.py` | Scores news headlines with FinBERT on the Kaggle GPU (used by `scripts/kaggle_news.py`) |
