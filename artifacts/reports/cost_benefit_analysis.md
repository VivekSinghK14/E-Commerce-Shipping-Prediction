# Cost and benefit analysis

The positive class is **Delivered**. Therefore a true negative is a correctly
identified delayed/returned shipment and a false positive is a missed risk.

| Classification outcome | Candidate count | Business treatment | Unit value |
|---|---:|---|---:|
| Correct risk flag (TN) | 303 | intervention opportunity | EUR 30.00 benefit |
| Correct on-time (TP) | 6721 | avoid unnecessary action | EUR 2.00 benefit |
| Missed risk (FP) | 1685 | delay/return not flagged | EUR 80.00 cost |
| Unnecessary intervention (FN) | 1291 | on-time order incorrectly flagged | EUR 10.00 cost |

- Candidate estimated net benefit: **EUR -125178.00**
- Baseline estimated net benefit: **EUR -143016.00**
- Candidate minus baseline: **EUR 17838.00**

These unit values are explicit assumptions for scenario comparison. Finance and
operations must replace them with verified values before a real investment decision.
