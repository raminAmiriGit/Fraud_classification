# Mistplay Fraud Demo Plan

## Demo flow (high level)
- Ingest/generate data: player events, device signals, and rewards/redemption behavior; persist as Delta tables.
- Build features: aggregate behavior windows (1h/24h/7d), device/account risk indicators, and reward velocity; register to Feature Store. The feature can come from different tables.
- Train + track model: supervised fraud classifier (e.g., LightGBM/LogReg) using MLflow tracking and register in MLflow Model Registry.
- Batch inference: score recent events/accounts and write results to a Delta table for analyst review.
- Real-time serving: deploy latest model to Model Serving and call endpoint with sample payloads.
- Monitor: basic data quality + drift checks (simple expectations on input features + prediction distribution snapshot).
## Key artifacts to build
- Notebook 1: 00_generate_data (synthetic generator or adapt public dataset).
- Notebook 2: 01_feature_pipeline (feature engineering, write to Feature Store).
- Notebook 3: 02_train_register (train, MLflow tracking, register model).
- Notebook 4: 03_batch_inference (score, write outputs, simple BI-ready table).
- Notebook 5: 04_model_serving_demo (deploy + sample requests).
## Implementation details
- Use a compact, fast synthetic dataset (~1M rows max. it can be even smaller) to fit quick runtime.
- Feature Store: create feature table keyed on account_id and device_id with time-windowed aggregates.
- MLflow: log metrics (AUC/PR), artifacts, and register the model as mistplay_fraud_model.
- Batch inference: schedule as a job with a small lookback window; write fraud_score and risk_bucket.
- Serving: use the latest Production stage model; include a request/response notebook snippet.
- Make the data simple and small. I don't need several hours of training. It can be few minutes of training. So data size can be small
- Data preferebly can be synthetized. However you can get idea from online datasets. Also I prefer to have several datasets and and have some aggregations, joins and clean up as well.
- Feature store is also important element which I want to emphasize benefit of it either in batch processing or model serving/ registery

