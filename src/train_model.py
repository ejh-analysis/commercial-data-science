"""
src/train_model.py
Executes SQL transformations via DuckDB and trains an explainable 
XGBoost churn classification model.
"""

import duckdb
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# 1. Connect DuckDB in-memory and execute SQL
con = duckdb.connect(database=":memory:")
con.execute("CREATE TABLE orders AS SELECT * FROM read_csv_auto('data/raw_orders.csv');")

with open("sql/01_customer_rfm.sql", "r") as f:
    query = f.read()

dataset = con.execute(query).df()
print(f"Loaded {len(dataset)} customer features directly from DuckDB SQL.\n")

# 2. Split Features & Target
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

# 3. Train Classifier
clf = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.05,
    eval_metric="logloss",
    random_state=42
)
clf.fit(X_train, y_train)

# 4. Evaluation
preds = clf.predict(X_test)
probs = clf.predict_proba(X_test)[:, 1]

print("=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, preds))
print(f"ROC-AUC Score: {roc_auc_score(y_test, probs):.4f}")