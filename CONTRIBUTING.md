# Contributing

## Ground rules

1. **No look-ahead.** A feature on date *t* may only use data public by the close of *t*. Add a test for every new feature.
2. **Never use 2025–26 data** before the final model freeze (`holdout_start` in `config/settings.yaml`).
3. **Every model is scored by `src/eval/metrics.py`** on the same weekly date grid.
4. **A feature group stays only if it improves rank IC** across folds (ablation), recorded in `docs/results.md`.
5. **`pytest -q` must pass** before every commit. GitHub Actions runs it on every push.
6. **Never commit secrets or data:** `.env`, `kaggle.json`, `data/`, `*.parquet`.
7. Settings go in `config/settings.yaml`, not hard-coded.

## Workflow

```
git checkout -b feature/timesfm          # one branch per change
# ... edit, then:
pytest -q
git add -A
git commit -m "Add TimesFM zero-shot runner"
git push -u origin feature/timesfm       # open a pull request on GitHub
```

Branch names: `feature/...`, `fix/...`, `exp/...` (experiments), `docs/...`.
Commit messages: imperative, one line, e.g. `Add sector-relative features`.

After any finished step, add a line to the work log in `docs/PROGRESS.md`.

## Reporting results

Use the **Experiment result** issue template, paste `metrics.csv` and the run folder name.
