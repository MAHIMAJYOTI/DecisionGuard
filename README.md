# DecisionGuard

A local credit and fraud decisioning project. The book is synthetic. Strategies are YAML, models are saved files, and a change from v1 to v2 is scored on the same rows before anyone ships it.

## Live demo

The dashboard reads a small snapshot in `app/data` plus the copied models in `app/models`. It does not open the warehouse, and it does not need DuckDB or dbt.

```powershell
streamlit run app/streamlit_app.py
```

Rebuild the snapshot after a model or strategy change:

```powershell
python -m src.demo.export
```

Screenshots (add the image files after a local run):

- `docs/demo/overview.png` — book size, pipeline, monthly applications
- `docs/demo/credit.png` — decision mix, decline reasons, calibration
- `docs/demo/try-it.png` — one application scored with the live engine
