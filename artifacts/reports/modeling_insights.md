# Modeling insights

- Selected candidate: `random_forest` (selected using validation ROC-AUC).
- Frozen validation threshold: `0.52`.
- Untouched-test ROC-AUC: `0.4991`; chance is `0.5000`.
- Candidate weighted F1: `0.6896`.
- Naive majority weighted F1: `0.7128`.
- Minority-class recall: `0.1524`.
- Estimated candidate net benefit: `EUR -125178.00`.
- Estimated baseline net benefit: `EUR -143016.00`.

The available features do not demonstrate reliable predictive signal. The result
supports the null hypothesis and does not justify model promotion.
