# Cost and benefit analysis

The positive class is **Delivered**. Therefore a true negative is a correctly
identified delayed shipment and a false positive is a missed delay.

| Classification outcome | Candidate count | Business treatment | Unit value |
|---|---:|---|---:|
| Correct delay flag (TN) | 712 | intervention opportunity | EUR 30.00 benefit |
| Correct delivered (TP) | 4097 | avoid unnecessary action | EUR 2.00 benefit |
| Missed delay (FP) | 789 | delayed shipment not flagged | EUR 80.00 cost |
| Unnecessary intervention (FN) | 3916 | delivered order incorrectly flagged as delayed | EUR 10.00 cost |

- Candidate estimated net benefit: **EUR -72726.00**
- Baseline estimated net benefit: **EUR -104054.00**
- Candidate minus baseline: **EUR 31328.00**

These unit values are explicit assumptions for scenario comparison. Finance and
operations must replace them with verified values before a real investment decision.
