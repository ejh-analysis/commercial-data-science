# Commercial-data-science
End-to-end commercial analytics pipeline using SQL and Python: cohort RFM segmentation, customer lifetime value, and churn risk classification.

# 📊 Commercial Customer Intelligence & Churn Engine

[![SQL](https://img.shields.io/badge/SQL-DuckDB-FFF000?style=flat&logo=duckdb&logoColor=black)](https://duckdb.org/)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-red?style=flat)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end commercial data science pipeline coupling in-process analytical SQL transformations (DuckDB) with class-balanced gradient boosted classification (XGBoost) and SHAP feature attribution.

---

## 📌 Technical Pipeline Architecture

```text
[Raw Transaction Logs (CSV)]
              │
              ▼
[DuckDB Analytical SQL Engine]
  ├── Multi-Stage CTEs (`sql/01_customer_rfm.sql`)
  ├── Window Functions (`LAG`, `NTILE(5)`)
  └── Behavioral Aggregations (Recency, Interval Dynamics, Monetary Tiers)
              │
              ▼
[Curated Feature Vector]
              │
              ▼
[XGBoost Classifier] ─── Cost-Sensitive Imbalance Correction (`scale_pos_weight`)
              │
              ▼
[SHAP Attribution Engine] ─── Local & Global Explainability (`TreeExplainer`)