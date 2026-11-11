"""Score the news history with FinBERT on Kaggle's free GPU, from your PC.

    python -m scripts.kaggle_news upload     send data/clean/news_articles.parquet as a private Kaggle dataset
    python -m scripts.kaggle_news run        start the FinBERT script on a Kaggle T4 GPU
    python -m scripts.kaggle_news status     is it still running?
    python -m scripts.kaggle_news download   fetch the scores -> data/clean/news_scored.parquet
    python -m scripts.kaggle_news all        upload, run, wait, download (leave it running)

Needs the Kaggle token in C:\\Users\\<you>\\.kaggle\\kaggle.json and a phone-verified Kaggle
account (required for GPU + internet). After the first upload, Kaggle can take a few minutes
to process the dataset before `run` can use it.
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from src.config import ROOT, path

DATASET_SLUG = "latticeiq-news"
KERNEL_SLUG = "latticeiq-finbert"


def kaggle_user() -> str:
    if os.getenv("KAGGLE_USERNAME"):
        return os.environ["KAGGLE_USERNAME"]
    f = Path.home() / ".kaggle" / "kaggle.json"
    if not f.exists():
        raise SystemExit(f"Kaggle token not found at {f}")
    return json.loads(f.read_text())["username"]


def kaggle(*args, check=True) -> subprocess.CompletedProcess:
    exe = shutil.which("kaggle") or str(Path(sys.executable).parent / "kaggle")
    r = subprocess.run([exe, *args], capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip()
    if out:
        print(out)
    if check and r.returncode != 0:
        raise SystemExit(f"kaggle {' '.join(args)} failed")
    return r


def upload() -> None:
    user = kaggle_user()
    src = path("clean") / "news_articles.parquet"
    if not src.exists():
        raise SystemExit("data/clean/news_articles.parquet missing. Run: python -m src.ingest.benzinga --combine")
    folder = ROOT / "data" / "kaggle_upload" / "news"
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, folder / "news_articles.parquet")
    (folder / "dataset-metadata.json").write_text(json.dumps({
        "title": DATASET_SLUG, "id": f"{user}/{DATASET_SLUG}", "licenses": [{"name": "other"}]}, indent=2))
    print(f"Uploading {src.stat().st_size / 1e6:.0f} MB to Kaggle as private dataset {user}/{DATASET_SLUG} ...")
    r = kaggle("datasets", "version", "-p", str(folder), "-m", "update", "-t", "-d", check=False)
    if r.returncode != 0 or "not found" in (r.stdout + r.stderr).lower():
        kaggle("datasets", "create", "-p", str(folder), "-t")
    print("Uploaded. Kaggle may need a few minutes to process it before `run`.")


def run() -> None:
    user = kaggle_user()
    folder = ROOT / "data" / "kaggle_upload" / "finbert_kernel"
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "kaggle" / "finbert" / "finbert_score.py", folder / "finbert_score.py")
    (folder / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{user}/{KERNEL_SLUG}", "title": KERNEL_SLUG, "code_file": "finbert_score.py",
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": True, "enable_internet": True,
        "dataset_sources": [f"{user}/{DATASET_SLUG}"], "competition_sources": [], "kernel_sources": [],
    }, indent=2))
    kaggle("kernels", "push", "-p", str(folder), "--accelerator", "NvidiaTeslaT4")
    print(f"Started. Watch it at https://www.kaggle.com/code/{user}/{KERNEL_SLUG}")


def status() -> str:
    r = kaggle("kernels", "status", f"{kaggle_user()}/{KERNEL_SLUG}", check=False)
    text = (r.stdout + r.stderr).lower()
    for s in ("complete", "error", "cancel", "running", "queued"):
        if s in text:
            return s
    return "unknown"


def download() -> None:
    out = ROOT / "data" / "kaggle_out" / "finbert"
    out.mkdir(parents=True, exist_ok=True)
    kaggle("kernels", "output", f"{kaggle_user()}/{KERNEL_SLUG}", "-p", str(out), "-o")
    f = out / "news_scored.parquet"
    if not f.exists():
        raise SystemExit("news_scored.parquet not in the Kaggle output. Check the run's log on kaggle.com.")
    new = pd.read_parquet(f)
    dest = path("clean") / "news_scored.parquet"
    if dest.exists():
        new = pd.concat([pd.read_parquet(dest), new]).drop_duplicates("id", keep="last")
    new.to_parquet(dest, index=False)
    print(f"Saved {len(new):,} scored articles -> {dest}. Mean score {new['score'].mean():.3f}")


def wait_and_download(poll_minutes: int = 3) -> None:
    while True:
        s = status()
        print(f"{time.strftime('%H:%M')} status: {s}")
        if s == "complete":
            return download()
        if s in ("error", "cancel"):
            raise SystemExit("Kaggle run failed. Open the link above, read the log, and send it to Claude.")
        time.sleep(poll_minutes * 60)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "upload":
        upload()
    elif cmd == "run":
        run()
    elif cmd == "status":
        print(status())
    elif cmd == "download":
        download()
    elif cmd == "all":
        upload()
        print("Giving Kaggle 3 minutes to process the dataset...")
        time.sleep(180)
        run()
        time.sleep(60)
        wait_and_download()
    else:
        print(__doc__)
