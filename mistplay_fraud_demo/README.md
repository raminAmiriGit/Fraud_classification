# Mistplay Fraud Detection Demo — Description, Goals, and Example

## What this demo is
A realistic, end‑to‑end fraud detection demo for a gaming company (Mistplay) built on Databricks. We simulate player activity, build behavioral and device‑risk features, train a fraud model, and then score users in batch and in real time.

## The problem we are solving
**Goal:** Detect and reduce reward abuse and account farming without harming legitimate players.

**Example scenario (fraud behavior):**
- A fraudster creates many new accounts from the same device or emulator.
- They repeatedly trigger reward claims (e.g., daily bonuses or referral rewards).
- They redeem rewards quickly, often using VPNs or mismatched IP locations.
- They show abnormal play patterns: very short sessions, high reward velocity, and repeated device reuse.

**Impact:** This causes inflated costs, distorted marketing attribution, and abuse of the rewards program.

## What data we use to detect it
We bring together multiple datasets so we can detect suspicious patterns through joins and aggregations:

### 1) Accounts table  
**What it captures:** account metadata and acquisition context  
**Example fields:**  
- `account_id`, `created_date`, `country`, `platform`, `marketing_channel`

### 2) Devices table  
**What it captures:** device fingerprint and risk indicators  
**Example fields:**  
- `device_id`, `device_type`, `os_version`, `is_emulator`, `device_risk_score`

### 3) Events table  
**What it captures:** gameplay and session behavior  
**Example fields:**  
- `event_ts`, `event_type`, `session_minutes`, `is_vpn`, `ip_country`

### 4) Rewards table  
**What it captures:** reward claims and redemptions  
**Example fields:**  
- `reward_ts`, `reward_type`, `reward_amount`, `redemption_channel`

---

## How the model detects fraud (features we engineer)
We aggregate behavior over short windows (1–7 days) and join across datasets to detect anomalies such as:

- **Reward velocity**: high number of rewards claimed in a short time  
- **Short sessions + high rewards**: minimal play but frequent reward activity  
- **Device reuse**: many accounts on the same device  
- **VPN usage rate**: high fraction of sessions from VPNs  
- **Country mismatch**: `country` vs. `ip_country` inconsistency  
- **Emulator + high rewards**: emulator usage correlates with abuse  

These features are stored in Feature Store so we can reuse them consistently during training and inference.

---

## Demo flow
1. **Generate synthetic data** for accounts, devices, events, and rewards.  
2. **Build features** (aggregations, joins, cleanup) and register in Feature Store.  
3. **Train and register a fraud model** with MLflow tracking.  
4. **Batch scoring** to produce a fraud review table.  
5. **Real‑time scoring** using Model Serving with only entity IDs.  

---

## Why this matters to the audience
- Shows how to detect abuse early and reduce financial loss.  
- Demonstrates a production‑ready ML lifecycle on Databricks.  
- Highlights governance and reuse via Feature Store and MLflow.  