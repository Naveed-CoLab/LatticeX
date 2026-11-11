"""FinBERT sentiment scoring. Used two ways:
  - on Kaggle's GPU for the full news history (kaggle/finbert/finbert_score.py is a copy of this logic)
  - on your PC's CPU for the daily update (a few hundred articles a day)

Local run (needs: pip install torch transformers):
    python -m src.news.sentiment                  scores data/clean/news_articles.parquet rows not yet scored

score = P(positive) - P(negative), from -1 (very negative) to +1 (very positive).
Model is set in config/settings.yaml (news.sentiment_model) so it can be swapped later:
ProsusAI/finbert is non-commercial (CC BY-NC-SA 4.0) -> fine for the FYP, replace before selling.
"""
import numpy as np
import pandas as pd

from src.config import load_settings, path


def article_text(df: pd.DataFrame) -> list:
    title = df["title"].fillna("")
    teaser = df["teaser"].fillna("")
    return (title + ". " + teaser).str.slice(0, 1000).tolist()


def score_texts(texts: list, model_name: str, batch_size: int | None = None, max_length: int = 128,
                device: str | None = None) -> pd.DataFrame:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = batch_size or (256 if device == "cuda" else 32)
    print(f"FinBERT on {device}, {len(texts):,} texts, batch {batch_size}", flush=True)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
    if device == "cuda":
        model = model.half()
    labels = {i: l.lower() for i, l in model.config.id2label.items()}
    out = np.zeros((len(texts), len(labels)), dtype=np.float32)
    order = np.argsort([len(t) for t in texts])          # similar lengths per batch = much faster
    with torch.no_grad():
        for k in range(0, len(order), batch_size):
            idx = order[k:k + batch_size]
            enc = tok([texts[i] for i in idx], padding=True, truncation=True, max_length=max_length,
                      return_tensors="pt").to(device)
            out[idx] = torch.softmax(model(**enc).logits.float(), dim=-1).cpu().numpy()
            if (k // batch_size) % 500 == 0:
                print(f"  {k + len(idx):,}/{len(texts):,}", flush=True)
    probs = pd.DataFrame(out, columns=[f"p_{labels[i]}" for i in range(len(labels))])
    probs["score"] = probs.get("p_positive", 0) - probs.get("p_negative", 0)
    return probs


def score_new_articles() -> None:
    s = load_settings()
    model_name = s.get("news", {}).get("sentiment_model", "ProsusAI/finbert")
    arts = pd.read_parquet(path("clean") / "news_articles.parquet")
    out_f = path("clean") / "news_scored.parquet"
    done = pd.read_parquet(out_f) if out_f.exists() else pd.DataFrame(columns=["id"])
    todo = arts[~arts["id"].isin(done["id"])]
    print(f"{len(arts):,} articles, {len(todo):,} not scored yet")
    if todo.empty:
        return
    probs = score_texts(article_text(todo), model_name)
    new = pd.concat([todo[["id"]].reset_index(drop=True), probs], axis=1)
    pd.concat([done, new], ignore_index=True).drop_duplicates("id", keep="last").to_parquet(out_f, index=False)
    print(f"Scored {len(new):,} -> {out_f}")


if __name__ == "__main__":
    score_new_articles()
