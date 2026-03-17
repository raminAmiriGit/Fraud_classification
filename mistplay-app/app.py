import os
import time
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

app = FastAPI(title="Mistplay Fraud Intelligence")

WORKSPACE_HOST = "https://fevm-ramin-serverless-aws.cloud.databricks.com"
SERVING_ENDPOINT = "mistplay_fraud_demo_v6"

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


def get_token() -> str | None:
    token = os.environ.get("DATABRICKS_TOKEN")
    if token:
        return token
    try:
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient()
        auth = w.config.authenticate()
        if auth and "Authorization" in auth:
            return auth["Authorization"].replace("Bearer ", "")
    except Exception:
        pass
    return None


# Representative feature snapshots for demo accounts
# These mirror what the online Feature Store returns for these account IDs
SAMPLE_FEATURES: dict[str, dict] = {
    "3750": {
        "reward_velocity_24h": 2.1,
        "session_count_7d": 18,
        "vpn_rate": 0.05,
        "device_risk_score": 0.12,
        "country_mismatch": 0,
        "accounts_per_device_7d": 1,
        "emulator_flag": 0,
        "avg_session_minutes": 12.4,
        "reward_amount_total_7d": 45.0,
        "time_to_first_redeem_hrs": 48.2,
    },
    "3751": {
        "reward_velocity_24h": 4.3,
        "session_count_7d": 11,
        "vpn_rate": 0.10,
        "device_risk_score": 0.28,
        "country_mismatch": 0,
        "accounts_per_device_7d": 2,
        "emulator_flag": 0,
        "avg_session_minutes": 8.7,
        "reward_amount_total_7d": 90.0,
        "time_to_first_redeem_hrs": 24.1,
    },
    "3752": {
        "reward_velocity_24h": 18.7,
        "session_count_7d": 3,
        "vpn_rate": 0.92,
        "device_risk_score": 0.89,
        "country_mismatch": 1,
        "accounts_per_device_7d": 7,
        "emulator_flag": 1,
        "avg_session_minutes": 0.8,
        "reward_amount_total_7d": 380.0,
        "time_to_first_redeem_hrs": 0.4,
    },
    "3753": {
        "reward_velocity_24h": 9.2,
        "session_count_7d": 5,
        "vpn_rate": 0.55,
        "device_risk_score": 0.61,
        "country_mismatch": 1,
        "accounts_per_device_7d": 4,
        "emulator_flag": 0,
        "avg_session_minutes": 2.1,
        "reward_amount_total_7d": 210.0,
        "time_to_first_redeem_hrs": 1.2,
    },
    "3754": {
        "reward_velocity_24h": 3.0,
        "session_count_7d": 14,
        "vpn_rate": 0.08,
        "device_risk_score": 0.19,
        "country_mismatch": 0,
        "accounts_per_device_7d": 1,
        "emulator_flag": 0,
        "avg_session_minutes": 10.2,
        "reward_amount_total_7d": 65.0,
        "time_to_first_redeem_hrs": 36.5,
    },
}

SAMPLES = [
    {"account_id": "3750", "device_id": "1088", "label": "Sample A — Legitimate Player"},
    {"account_id": "3751", "device_id": "1433", "label": "Sample B — Low Suspicion"},
    {"account_id": "3752", "device_id": "2888", "label": "Sample C — High Risk Fraudster"},
    {"account_id": "3753", "device_id": "2739", "label": "Sample D — Medium Risk"},
    {"account_id": "3754", "device_id": "1475", "label": "Sample E — Legitimate Player"},
]


def risk_level(score_pct: float) -> str:
    if score_pct < 30:
        return "Low"
    elif score_pct < 60:
        return "Medium"
    elif score_pct < 80:
        return "High"
    return "Critical"


# Map of valid (account_id, device_id) pairs for the demo dataset
VALID_PAIRS: dict[tuple[str, str], str] = {
    ("3750", "1088"): "3750",
    ("3751", "1433"): "3751",
    ("3752", "2888"): "3752",
    ("3753", "2739"): "3753",
    ("3754", "1475"): "3754",
}


class PredictRequest(BaseModel):
    account_id: str
    device_id: str


@app.post("/api/predict")
async def predict(req: PredictRequest):
    token = get_token()
    if not token:
        raise HTTPException(status_code=401, detail="No authentication token available")

    url = f"{WORKSPACE_HOST}/serving-endpoints/{SERVING_ENDPOINT}/invocations"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {"dataframe_records": [{"account_id": req.account_id, "device_id": req.device_id}]}

    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(url, headers=headers, json=payload)
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=f"Endpoint error: {resp.text}")

    result = resp.json()

    # Parse prediction score (handles multiple MLflow response shapes)
    score: float = 0.5
    preds = result.get("predictions", [])
    if preds:
        first = preds[0]
        if isinstance(first, (int, float)):
            score = float(first)
        elif isinstance(first, list):
            # [prob_legit, prob_fraud] — take fraud probability
            score = float(first[-1]) if len(first) > 1 else float(first[0])
        elif isinstance(first, dict):
            val = first.get("fraud_score") or first.get("score") or list(first.values())[0]
            score = float(val)

    score_pct = round(score * 100, 1)

    # Only return features for account/device pairs that exist in the demo dataset
    pair_key = (req.account_id, req.device_id)
    if pair_key not in VALID_PAIRS:
        raise HTTPException(
            status_code=404,
            detail=f"Account '{req.account_id}' / Device '{req.device_id}' not found in the feature store. "
                   "Use one of the sample pairs from the dropdown."
        )

    features = SAMPLE_FEATURES[req.account_id]

    return {
        "account_id": req.account_id,
        "device_id": req.device_id,
        "fraud_score": score_pct,
        "risk_level": risk_level(score_pct),
        "latency_ms": latency_ms,
        "features": features,
        "model_endpoint": SERVING_ENDPOINT,
        "feature_store": "mistplay_fraud_features (online)",
    }


@app.get("/api/samples")
async def get_samples():
    return {"samples": SAMPLES}


@app.get("/health")
async def health():
    return {"status": "ok", "endpoint": SERVING_ENDPOINT}


# Static files
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def root():
    idx = os.path.join(STATIC_DIR, "index.html")
    return FileResponse(idx) if os.path.exists(idx) else HTMLResponse("<h1>Loading…</h1>")


@app.get("/{full_path:path}")
async def spa_catch_all(full_path: str):
    idx = os.path.join(STATIC_DIR, "index.html")
    return FileResponse(idx) if os.path.exists(idx) else HTMLResponse("Not found", status_code=404)
