# Customer Churn Prediction — Kaggle Playground Series S6E3

Predicting customer churn for a telecommunications company using a synthetic version of the IBM Telco Customer Churn dataset.

## Approach

**KNN-Anchored Feature Engineering** — Each synthetic training row is matched to its 5 nearest neighbours in the original IBM dataset (by tenure, MonthlyCharges, TotalCharges). The average churn rate of those real customers is used as a ground-truth signal, giving the model a bridge between synthetic and real-world behaviour.

**Engineered Features** — Beyond the KNN signal, the notebook builds features capturing contract risk (month-to-month fibre optic customers, electronic check payment), customer protection level, billing patterns (charges per month, deviation from expected total), and tenure-based signals (early-life flag, tenure squared).

**Optuna Hyperparameter Tuning** — XGBoost and LightGBM are tuned over 15 trials each with early stopping, searching over learning rate, tree depth, regularisation, subsampling, and leaf parameters.

**Stacking Ensemble** — The tuned XGBoost and LightGBM models are combined via a StackingClassifier with a Logistic Regression meta-learner trained on 5-fold out-of-fold predictions.

## Results

| Model | AUC |
|-------|-----|
| LightGBM (default) | 0.9159 |
| XGBoost (default) | 0.9160 |
| CatBoost (default) | 0.9156 |
| XGBoost (Optuna-tuned) | 0.9191 |
| Stacking Ensemble | TBD |

## Tech Stack

Python, pandas, NumPy, scikit-learn, LightGBM, XGBoost, CatBoost, Optuna

## Files

- `ibm-customer-churn.ipynb` — Full notebook with EDA, feature engineering, tuning, and submission

## Links

- [Kaggle Competition](https://www.kaggle.com/competitions/playground-series-s6e3)
- [Kaggle Notebook](https://www.kaggle.com/amiraslebik) *(to be updated with direct link once published)*
