# Business objective assessment

## Pros

- The pipeline is reproducible and the test set is isolated from selection.
- The chosen threshold identifies some delayed shipments.
- All business assumptions and failed gates are explicit and auditable.

## Cons

- ROC-AUC is approximately chance, so ranking is unreliable.
- The candidate does not improve weighted F1 over the majority baseline.
- Minority recall and economic value depend heavily on threshold and assumed costs.
- Current fields contain little pre-outcome signal; deployment could create cost without benefit.

## Conclusion

**RETURN_TO_BUSINESS_UNDERSTANDING** — Revisit target, available predictors, and business framing; do not deploy.
