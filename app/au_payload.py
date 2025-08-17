# app/au_payload.py
from __future__ import annotations
from pathlib import Path
import pandas as pd

def build_au_payload(session_csv_path: str | Path, mode: str = "latest", n: int = 40) -> dict:
    """
    Returns a small dict ready to send to the LLM.
    mode: "latest" (last row) or "window" (mean of last n rows)
    """
    p = Path(session_csv_path)
    if not p.exists():
        return {"_ok": False, "_err": f"CSV not found: {p}"}

    df = pd.read_csv(p, parse_dates=["ts"], engine="python", on_bad_lines="skip")
    if df.empty:
        return {"_ok": False, "_err": "CSV is empty"}

    # Ensure numeric for AU cols + proxies if present
    au_cols = [c for c in df.columns if c.startswith("AU")]
    for c in au_cols + ["valence_proxy", "arousal_proxy"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    if mode == "latest":
        row = df.tail(1).to_dict(orient="records")[0]
        payload = {k: row[k] for k in row if k.startswith("AU") or k in ["ts","valence_proxy","arousal_proxy","expr","expr_score"]}
        payload["_mode"] = "single_pulse"
        payload["_ok"] = True
        return payload

    # window mode (mean of last n pulses)
    win = df.tail(int(n)).copy()
    means = {c: float(win[c].mean()) for c in win.columns if c.startswith("AU") or c in ["valence_proxy","arousal_proxy"]}
    payload = {
        "_mode": "window_means",
        "_ok": True,
        "count": len(win),
        "ts_start": str(win["ts"].iloc[0]) if "ts" in win else None,
        "ts_end":   str(win["ts"].iloc[-1]) if "ts" in win else None,
        "means": means,
    }
    return payload
