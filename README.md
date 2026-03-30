# Churn: KNN Anchoring + Optuna-Tuned Ensembles | Top 22%

Kaggle Playground Series S6E3 — Predicting customer churn for a telecommunications company using a synthetic version of the IBM Telco Customer Churn dataset.

**Leaderboard: 821st / 3,773 teams (Top 22%)**

## Approach

### KNN-Anchored Feature Engineering
The competition dataset is synthetic, so some labels may not reflect real-world behaviour. To bridge this gap, each synthetic row is matched to its 5 nearest neighbours in the original IBM Telco dataset (by tenure, MonthlyCharges, TotalCharges). The average churn rate of those real customers is used as a ground-truth signal, giving the model a direct link between synthetic and real-world churn patterns.

### Feature Engineering
Beyond the KNN signal, the notebook engineers features across several dimensions:

- **Contract risk:** month-to-month fibre optic flag, electronic check payment flag
- **Customer protection:** count of missing security/support add-ons
- **Billing patterns:** charges per month, deviation from expected total, monthly-to-total ratio
- **Tenure signals:** early-life flag (first 6 months), tenure squared, tenure bin interactions
- **Original dataset priors:** per-category churn rates mapped from the real IBM data

### Optuna Hyperparameter Tuning
XGBoost and LightGBM are each tuned over 15 trials with early stopping, searching over learning rate, tree depth, regularisation, subsampling, and leaf parameters.

### Ensemble Strategy
Two ensemble approaches were tested:

- **Stacking:** StackingClassifier with Logistic Regression meta-learner on 5-fold out-of-fold predictions
- **Rank Averaging:** Rank-based blending of predicted probabilities from both models

Rank averaging outperformed stacking on the leaderboard and was used as the final submission.

## Results

| Model | Validation AUC | Leaderboard AUC |
|-------|---------------|-----------------|
| LightGBM (default) | 0.9179 | - |
| XGBoost (default) | 0.9181 | - |
| XGBoost (Optuna-tuned) | 0.9191 | - |
| Stacking Ensemble (tuned) | 0.9255 | 0.91433 |
| Rank Averaging (tuned) | 0.9259 | 0.91438 |

## Tech Stack

Python, pandas, NumPy, scikit-learn, LightGBM, XGBoost, Optuna, SciPy, Matplotlib, Seaborn

## Files

- `churn-knn-optuna-ensembles-top-22.ipynb` — Full notebook with EDA, feature engineering, tuning, and submission

## Links

- [Kaggle Competition](https://www.kaggle.com/competitions/playground-series-s6e3)
- [Kaggle Notebook](https://www.kaggle.com/code/amiraslebik/churn-knn-optuna-ensembles-top-21)
