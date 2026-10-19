# Evaluation report

## Assessment

The model and threshold were frozen using validation data. The test split was
then used once for final assessment.

- Candidate: `xgboost`
- Final selected algorithm: `xgboost`
- Approved deployment model: `NONE`
- Test ROC-AUC: `0.4890`
- Test weighted F1: `0.5717`
- Baseline weighted F1: `0.7701`
- Test minority recall: `0.4744`
- Candidate net benefit: `EUR -72726.00`
- Baseline net benefit: `EUR -104054.00`
- Failed business gates: `roc_auc, weighted_f1_improvement_over_baseline, candidate_net_benefit_eur`

## Checkpoint decision

**RETURN_TO_BUSINESS_UNDERSTANDING**

Revisit target, available predictors, and business framing; do not deploy.
