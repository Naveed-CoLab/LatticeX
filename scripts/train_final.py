"""Train the model for daily use and save it (once; repeat every month or quarter).

    python -m scripts.train_final                    all three horizons, exactly the tested v1.0 recipe
    python -m scripts.train_final --horizons 20      one horizon
    python -m scripts.kaggle_cloud final-model       the same on Kaggle's GPU (about 15 minutes); the model file is
                                                     copied to models\\ on your PC when it finishes

Needs PyTorch (Kaggle has it; on a PC: pip install torch). Scoring afterwards does not need it.
See src/models/production.py.
"""
import argparse

from src.models.production import train

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", type=int, nargs="+")
    ap.add_argument("--train-step", type=int, default=1, help="use every Nth day (1 = the tested recipe; 5 = quick trial)")
    ap.add_argument("--since", help="use prices from this date only, e.g. 2016-01-01 (quick trial; the full run needs about 8 GB of memory)")
    a = ap.parse_args()
    train(a.horizons, a.train_step, since=a.since)
