---
title: Shipping Delay Predictor
emoji: 📦
colorFrom: blue
colorTo: indigo
sdk: static
app_file: index.html
pinned: false
short_description: Predicts Delivered vs Delayed e-commerce orders (MLOps demo)
---

# E-Commerce Shipping Delay Predictor

Course prototype for *Development and Operationalization of Data Science
Solutions* (MLOps-DSPM). The model predicts whether an order will be
**Delivered** or **Delayed** from details known when the order is placed
(region, product category, shipping mode, shipping cost, order date).
Returned orders are out of scope and were removed before training.

The trained XGBoost model runs in the browser (`predictor.js` + `model.json`).
Its predictions are identical to the project's FastAPI service, verified on
all 47,567 in-scope orders.

**Evaluation.** On the untouched test set (9,514 orders) the model reaches a
delayed-order recall of 0.474 and a lower estimated cost than the naive
baseline, while ROC-AUC is 0.489: the order-time features do not yet separate
delayed from delivered orders reliably. The next iteration adds pre-shipment
operational data (carrier, warehouse, route, promised delivery date). The app
is published as a live prototype of the complete MLOps pipeline.
