"""
src/pipeline.py

Reusable training + fairness evaluation pipeline for the thesis.
Used across all datasets (German Credit, Taiwan Credit Default, Folktables
ACS Employment) and all experiment stages (baseline, 3-subset design,
Reweighing mitigation).
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from aif360.metrics import ClassificationMetric


models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42)
}


def train_and_evaluate(dataset, model, protected_attr, scale=False, seed=42):
    """
    Train a single model on an AIF360 dataset and compute accuracy +
    fairness metrics against a binary protected attribute.

    Handles datasets whose favorable/unfavorable labels aren't already
    coded as 0/1 (e.g. German Credit uses 1=good, 2=bad) — some models
    (XGBoost in particular) require labels to be exactly 0 and 1.
    """
    train, test = dataset.split([0.7], shuffle=True, seed=seed)
    X_train, y_train_raw = train.features, train.labels.ravel()
    X_test, y_test_raw = test.features, test.labels.ravel()

    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # --- label remapping (fixes non-0/1 label coding) ---
    fav_label = dataset.favorable_label
    unfav_label = dataset.unfavorable_label

    y_train = np.where(y_train_raw == fav_label, 1, 0)
    y_test = np.where(y_test_raw == fav_label, 1, 0)

    model.fit(X_train, y_train)
    preds = model.predict(X_test)  # 0/1

    # map predictions back to the dataset's native label coding so
    # AIF360's ClassificationMetric (which reads favorable_label off
    # the dataset objects) stays consistent
    preds_native = np.where(preds == 1, fav_label, unfav_label)

    test_pred = test.copy()
    test_pred.labels = preds_native.reshape(-1, 1)

    metric = ClassificationMetric(
        test, test_pred,
        unprivileged_groups=[{protected_attr: 0}],
        privileged_groups=[{protected_attr: 1}]
    )

    importances = None
    if hasattr(model, 'feature_importances_'):
        importances = pd.Series(
            model.feature_importances_, index=train.feature_names
        ).sort_values(key=abs, ascending=False)
    elif hasattr(model, 'coef_'):
        importances = pd.Series(
            model.coef_[0], index=train.feature_names
        ).sort_values(key=abs, ascending=False)

    metrics = {
        'accuracy': (preds == y_test).mean(),
        # Did both groups receive positive predictions at the same rate?
        'stat_parity_diff': metric.statistical_parity_difference(),
        # Among people who truly deserved a positive prediction, did both groups have the same chance of getting one?
        'equal_opp_diff': metric.equal_opportunity_difference(),
        # Same idea as statistical parity, but as a ratio instead of a difference
        'disparate_impact': metric.disparate_impact(),
        # Compares both false positive and true positive rate gaps — a stricter, two-sided check
        'equalized_odds_diff': metric.average_odds_difference()
    }

    return metrics, importances


def run_all_models(dataset, protected_attr, models_dict=None, seed=42):
    """
    Convenience wrapper: runs train_and_evaluate for every model in
    models_dict (defaults to the module-level `models`), returns a
    results DataFrame plus a dict of feature-importance Series.
    """
    if models_dict is None:
        models_dict = models

    results_table = []
    importance_dict = {}

    for name, model in models_dict.items():
        scale = (name == 'Logistic Regression')
        metrics, importances = train_and_evaluate(
            dataset, model, protected_attr, scale=scale, seed=seed
        )
        metrics['model'] = name
        results_table.append(metrics)
        if importances is not None:
            importance_dict[name] = importances

    results_df = pd.DataFrame(results_table).set_index('model')
    return results_df, importance_dict