# LatticeIQ: score Benzinga news with FinBERT on Kaggle's GPU.
# Pushed and started by: python -m scripts.kaggle_news run
# Reads:  the private dataset <you>/latticeiq-news (news_articles.parquet)
# Writes: /kaggle/working/news_scored.parquet  (id, p_positive, p_negative, p_neutral, score)
# Self-contained on purpose: Kaggle runs this single file.
import glob
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL = "ProsusAI/finbert"
BATCH, MAX_LEN = 256, 128

files = glob.glob("/kaggle/input/**/news_articles.parquet", recursive=True)
assert files, "news_articles.parquet not found under /kaggle/input. Is the dataset attached?"
arts = pd.read_parquet(files[0], columns=["id", "title", "teaser"])
texts = (arts["title"].fillna("") + ". " + arts["teaser"].fillna("")).str.slice(0, 1000).tolist()
print(f"{len(texts):,} articles from {files[0]}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, torch.cuda.get_device_name(0) if device == "cuda" else "")
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(device).eval()
if device == "cuda":
    model = model.half()
labels = {i: l.lower() for i, l in model.config.id2label.items()}

# sort by length so each batch has similar-length texts (much faster)
order = np.argsort([len(t) for t in texts])
probs = np.zeros((len(texts), len(labels)), dtype=np.float32)
t0 = time.time()
with torch.no_grad():
    for k in range(0, len(order), BATCH):
        idx = order[k:k + BATCH]
        enc = tok([texts[i] for i in idx], padding=True, truncation=True, max_length=MAX_LEN,
                  return_tensors="pt").to(device)
        probs[idx] = torch.softmax(model(**enc).logits.float(), dim=-1).cpu().numpy()
        if (k // BATCH) % 200 == 0:
            done = k + len(idx)
            print(f"{done:,}/{len(texts):,}  {done / max(time.time() - t0, 1e-9):.0f} articles/s")

out = pd.DataFrame(probs, columns=[f"p_{labels[i]}" for i in range(len(labels))]).astype("float32")
out.insert(0, "id", arts["id"].values)
out["score"] = (out["p_positive"] - out["p_negative"]).astype("float32")
out.to_parquet("/kaggle/working/news_scored.parquet", index=False)
print(f"done: {len(out):,} scored in {time.time() - t0:.0f}s. Mean score {out['score'].mean():.3f}")
