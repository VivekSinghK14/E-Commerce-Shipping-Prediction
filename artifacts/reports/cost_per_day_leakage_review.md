# Cost_per_Day leakage review

**Decision: EXCLUDED FROM ALL MODEL SELECTION AND EVALUATION.**

Delivery_Days is only known after delivery, so Cost_per_Day is unavailable at the order-time prediction point and would leak future outcome information.

A high feature importance or apparent score improvement would not be valid evidence; it would measure access to post-outcome information.
