-- sql/01_customer_rfm.sql
-- Multi-stage CTE feature extraction for customer retention scoring

WITH order_intervals AS (
    SELECT 
        customer_id,
        CAST(order_date AS DATE) AS order_date,
        order_amount,
        LAG(CAST(order_date AS DATE)) OVER (
            PARTITION BY customer_id 
            ORDER BY CAST(order_date AS DATE)
        ) AS prev_order_date
    FROM orders
),

customer_aggregates AS (
    SELECT 
        customer_id,
        COUNT(*) AS total_orders,
        ROUND(SUM(order_amount), 2) AS monetary_value,
        ROUND(AVG(order_amount), 2) AS avg_basket_size,
        DATE_DIFF('day', MAX(order_date), DATE '2026-01-01') AS recency_days,
        ROUND(AVG(DATE_DIFF('day', prev_order_date, order_date)), 1) AS avg_reorder_interval_days
    FROM order_intervals
    GROUP BY customer_id
)

SELECT 
    customer_id,
    recency_days,
    total_orders,
    monetary_value,
    avg_basket_size,
    COALESCE(avg_reorder_interval_days, 0) AS avg_reorder_interval_days,
    NTILE(5) OVER (ORDER BY monetary_value DESC) AS monetary_tier,
    CASE 
        WHEN recency_days > 120 THEN 1 
        ELSE 0 
    END AS is_churned
FROM customer_aggregates;