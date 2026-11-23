# AquaWatch V3 — HCI-oriented FYP prototype

This version replaces the older basic Streamlit interface with an operator-oriented overview, guided MP4 analysis, a colour-coded segment timeline, selected segment playback, searchable history, and CSV exports.

## Setup

1. Keep the `models/fleeing_vs_combined.joblib` model and `weights/best_combine.pt` / `weights/bestv7.pt` files in the project.
2. Open a terminal inside `AquaWatch_V3`.
3. Install dependencies: `python -m pip install -r requirements.txt`
4. Launch on a dedicated port: `python -m streamlit run app.py --server.port 8503`
5. Open `http://localhost:8503`.

The source-only archive can be extracted first; the model and YOLO weight archives from the earlier delivery may be merged into `models/` and `weights/` respectively.

## Scope and caveats

- This is a prototype for **uploaded recordings**, not real-time monitoring or deployed pond surveillance.
- The trained Random Forest and YOLO weights are unchanged.
- The interface shows segment-level predictions and saves records in `aquawatch.sqlite3`.
- The model's training-time preprocessing and feature scaling must be validated against inference before interpreting predictions as reliable.
- Full end-to-end MP4 testing requires the installed Python dependencies and representative test footage. This package was syntax-checked, not empirically validated on videos.
