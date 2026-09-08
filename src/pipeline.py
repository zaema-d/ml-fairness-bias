"""
src/pipeline.py

Reusable training + fairness evaluation pipeline
Used across all datasets (German Credit, Taiwan Credit Default, Folktables
ACS Employment) and all experiment stages (baseline, 3-subset design,
Reweighing mitigation).
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from aif360.metrics import ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing


models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42)
}


def _fit_and_score(X_train, y_train_raw, test, model, protected_attr,
                    fav_label, unfav_label, feature_names, scale=False,
                    sample_weight=None):
    
    # Train one model and evaluate it on a held-out test set.
    # This helper works with raw arrays instead of a full AIF360 dataset,
    # which makes it easier to create subsets, masks, or reweighted samples.
    # The test set stays as a full AIF360 dataset because fairness metrics need
    # the original protected-attribute structure.

    X_test, y_test_raw = test.features, test.labels.ravel()

    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # Convert labels to a simple 0/1 format for sklearn.
    # 1 means the favorable outcome, 0 means the unfavorable one.

    y_train = np.where(y_train_raw == fav_label, 1, 0)
    y_test = np.where(y_test_raw == fav_label, 1, 0)

    if sample_weight is not None:
        model.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # Convert predictions back to the dataset's original label encoding
    # so AIF360 can evaluate fairness correctly.

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
            model.feature_importances_, index=feature_names
        ).sort_values(key=abs, ascending=False)
    elif hasattr(model, 'coef_'):
        importances = pd.Series(
            model.coef_[0], index=feature_names
        ).sort_values(key=abs, ascending=False)

    metrics = {
        'accuracy': (preds == y_test).mean(),
        'stat_parity_diff': metric.statistical_parity_difference(),
        'equal_opp_diff': metric.equal_opportunity_difference(),
        'disparate_impact': metric.disparate_impact(),
        'equalized_odds_diff': metric.average_odds_difference()
    }

    return metrics, importances


def train_and_evaluate(dataset, model, protected_attr, scale=False, seed=42):

    # Train one model on a 70/30 split and evaluate it on the held-out test set.

    train, test = dataset.split([0.7], shuffle=True, seed=seed)
    X_train, y_train_raw = train.features, train.labels.ravel()

    return _fit_and_score(
        X_train, y_train_raw, test, model, protected_attr,
        fav_label=test.favorable_label, unfav_label=test.unfavorable_label,
        feature_names=train.feature_names, scale=scale
    )


def run_all_models(dataset, protected_attr, models_dict=None, seed=42):

    # Run every model in the model dictionary on the same dataset
    # and return one results table plus feature importance for each model.
  
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


def run_subset_experiment(dataset, protected_attr, models_dict=None, seed=42):

    # Compare three training setups:
    # 1) privileged-only
    # 2) unprivileged-only
    # 3) combined data

    # All three are evaluated on the same test set.

    if models_dict is None:
        models_dict = models

    train, test = dataset.split([0.7], shuffle=True, seed=seed)

    X_train_full = train.features
    y_train_full_raw = train.labels.ravel()

    # Pick the correct protected-attribute column by name.
    # This matters because some datasets have more than one protected feature.
    attr_col = train.protected_attribute_names.index(protected_attr)
    protected_vals = train.protected_attributes[:, attr_col]

    # Create clean subsets using boolean masks.
    # This is simpler and more reliable than using AIF360 subset logic here.

    fav_label = test.favorable_label
    unfav_label = test.unfavorable_label
    feature_names = train.feature_names

    priv_mask = protected_vals == 1
    unpriv_mask = protected_vals == 0

    subsets = {
        'privileged_only': (X_train_full[priv_mask], y_train_full_raw[priv_mask]),
        'unprivileged_only': (X_train_full[unpriv_mask], y_train_full_raw[unpriv_mask]),
        'combined': (X_train_full, y_train_full_raw)
    }

    results_table = []

    for subset_name, (X_sub, y_sub_raw) in subsets.items():
        for model_name, model in models_dict.items():
            model_instance = clone(model)  # fresh, unfitted copy each time
            scale = (model_name == 'Logistic Regression')

            metrics, _ = _fit_and_score(
                X_sub, y_sub_raw, test, model_instance, protected_attr,
                fav_label=fav_label, unfav_label=unfav_label,
                feature_names=feature_names, scale=scale
            )
            metrics['training_subset'] = subset_name
            metrics['model'] = model_name
            results_table.append(metrics)

    results_df = pd.DataFrame(results_table).set_index(['training_subset', 'model'])
    return results_df


def run_reweighing_experiment(dataset, protected_attr, models_dict=None, seed=42):
    """
    Apply AIF360's Reweighing method to the training data so
    the model learns from a more balanced distribution.
    Then evaluate each model on the same held-out test set.
    """
    if models_dict is None:
        models_dict = models

    train, test = dataset.split([0.7], shuffle=True, seed=seed)

    rw = Reweighing(
        unprivileged_groups=[{protected_attr: 0}],
        privileged_groups=[{protected_attr: 1}]
    )
    train_rw = rw.fit_transform(train)

    X_train = train_rw.features
    y_train_raw = train_rw.labels.ravel()
    sample_weight = train_rw.instance_weights

    fav_label = test.favorable_label
    unfav_label = test.unfavorable_label
    feature_names = train_rw.feature_names

    results_table = []

    for model_name, model in models_dict.items():
        model_instance = clone(model)
        scale = (model_name == 'Logistic Regression')

        metrics, _ = _fit_and_score(
            X_train, y_train_raw, test, model_instance, protected_attr,
            fav_label=fav_label, unfav_label=unfav_label,
            feature_names=feature_names, scale=scale,
            sample_weight=sample_weight
        )
        metrics['model'] = model_name
        results_table.append(metrics)

    results_df = pd.DataFrame(results_table).set_index('model')
    return results_df