"""
src/generate_data.py
Generates a realistic multi-year transaction log for customer lifetime 
value and churn prediction modeling.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)
n_customers = 1500
records = []

start_date = datetime(2024, 1, 1)

for customer_id in range(1001, 1001 + n_customers):
    # Determine customer engagement profile
    churn_bias = np.random.rand()
    n_orders = np.random.poisson(lam=4 if churn_bias > 0.4 else 1)
    
    current_date = start_date + timedelta(days=int(np.random.uniform(0, 400)))
    
    for _ in range(max(1, n_orders)):
        amount = np.random.exponential(scale=65.0) + 12.50
        records.append({
            "order_id": f"ORD-{len(records)+10000}",
            "customer_id": f"CUST-{customer_id}",
            "order_date": current_date.strftime("%Y-%m-%d"),
            "order_amount": round(amount, 2)
        })
        # Space subsequent orders over time
        current_date += timedelta(days=int(np.random.exponential(scale=45) + 5))

df = pd.DataFrame(records)
df.to_csv("data/raw_orders.csv", index=False)
print(f"Generated {len(df)} transactions across {n_customers} customers -> data/raw_orders.csv")