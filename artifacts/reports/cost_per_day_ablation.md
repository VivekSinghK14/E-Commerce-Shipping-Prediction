# Cost_per_Day feature ablation

`Cost_per_Day = Shipping_Cost / Delivery_Days` was tested using the same
`random_forest` algorithm, seed, training rows, and validation rows.

- ROC-AUC with feature: `0.497992`
- ROC-AUC without feature: `0.495474`
- Delta (with − without): `0.002518`
- Feature importance/coefficient: `0.25347379539687265`
- Material improvement (>= 0.01 ROC-AUC and overall gate passed): `False`

The delta, not feature importance alone, answers whether it improves validated
performance. Importance describes model usage, not causality. Since the feature
is derived from `Shipping_Cost` and `Delivery_Days`, it may be redundant.

**Conclusion:** Cost_per_Day does not provide a material validated improvement; keep only if its business interpretability is useful.
