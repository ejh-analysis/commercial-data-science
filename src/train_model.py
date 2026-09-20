"""
src/train_model.py
Executes SQL transformations via DuckDB, trains a class-balanced XGBoost 
classifier, and generates SHAP explainability plots.
"""

import os
import duckdb
import matplotlib.pyplot as plt
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

# 1. Connect DuckDB in-memory and execute SQL
con = duckdb.connect(database=":memory:")
con.execute("CREATE TABLE orders AS SELECT * FROM read_csv_auto('data/raw_orders.csv');")

with open("sql/01_customer_rfm.sql", "r") as f:
    query = f.read()

dataset = con.execute(query).df()
print(f"Loaded {len(dataset)} customer features directly from DuckDB SQL.\n")

# 2. Features & Target Split
feature_cols = [
    "total_orders", 
    "monetary_value", 
    "avg_basket_size", 
    "avg_reorder_interval_days", 
    "monetary_tier"
]
X = dataset[feature_cols]
y = dataset["is_churned"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 3. Handle Imbalance: Calculate scale_pos_weight
count_negative = (y_train == 0).sum()
count_positive = (y_train == 1).sum()
scale_pos_weight = count_negative / max(1, count_positive)

clf = xgb.XGBClassifier(
    n_estimators=120,
    max_depth=4,
    learning_rate=0.05,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    random_state=42
)
clf.fit(X_train, y_train)

# 4. Evaluation
preds = clf.predict(X_test)
probs = clf.predict_proba(X_test)[:, 1]

print("=== CLASSIFICATION REPORT (CLASS BALANCED) ===")
print(classification_report(y_test, preds))
print(f"ROC-AUC Score: {roc_auc_score(y_test, probs):.4f}\n")

# 5. Generate & Save SHAP Explainability Plot
os.makedirs("assets", exist_ok=True)
explainer = shap.TreeExplainer(clf)
shap_values = explainer(X_test)

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, show=False)
plt.title("Customer Churn Risk: SHAP Feature Attribution", fontsize=14, pad=15)
plt.tight_layout()
plt.savefig("assets/shap_summary.png", dpi=300)
plt.close()
print("Saved feature attribution plot -> assets/shap_summary.png")