# Evaluation report

## Assessment

The model and threshold were frozen using validation data. The test split was
then used once for final assessment.

- Candidate: `random_forest`
- Test ROC-AUC: `0.4991`
- Test weighted F1: `0.6896`
- Baseline weighted F1: `0.7128`
- Test minority recall: `0.1524`
- Candidate net benefit: `EUR -125178.00`
- Baseline net benefit: `EUR -143016.00`
- Failed business gates: `roc_auc, minority_recall, weighted_f1_improvement_over_baseline, candidate_net_benefit_eur`

## Checkpoint decision

**RETURN_TO_BUSINESS_UNDERSTANDING**

Revisit target, available predictors, and business framing; do not deploy.
