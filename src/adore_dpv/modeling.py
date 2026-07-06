from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC


RANDOM_SEED = 42

FEATURE_COLUMNS = [
    "miR92a_Delta_I",
    "miR21_Delta_I",
    "miR92a_Ep",
    "miR21_Ep",
    "miR92a_Ah",
    "miR21_Ah",
    "miR92a_FWHM",
    "miR21_FWHM",
]

REQUIRED_COLUMNS = ["sample_id", "group", *FEATURE_COLUMNS]

LABEL_MAP = {
    "Healthy": 0,
    "Disease": 1,
}


@dataclass(frozen=True)
class EvaluationConfig:
    n_splits: int = 5
    n_repeats: int = 5
    inner_splits: int = 3
    n_trials: int = 20
    random_state: int = RANDOM_SEED
    tune: bool = True


def validate_feature_table(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    labels = sorted(set(df["group"].astype(str)))
    supported = sorted(LABEL_MAP)
    extra = [label for label in labels if label not in LABEL_MAP]
    if extra:
        raise ValueError(f"Supported group labels are {supported}; received {extra}")

    if df[FEATURE_COLUMNS].isna().any().any():
        raise ValueError("Feature columns contain missing values")

    for column in FEATURE_COLUMNS:
        pd.to_numeric(df[column], errors="raise")


def load_feature_table(path: str | Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    validate_feature_table(df)
    X = df[FEATURE_COLUMNS].astype(float).to_numpy()
    y = df["group"].map(LABEL_MAP).to_numpy()
    sample_ids = df["sample_id"].astype(str).to_numpy()
    return df, X, y, sample_ids


def _select_first_column(X: np.ndarray) -> np.ndarray:
    return X[:, [0]]


def build_trial_model(model_name: str, trial: Any | None, random_state: int = RANDOM_SEED):
    if model_name == "svm":
        c = trial.suggest_float("C", 0.1, 100, log=True) if trial else 1.0
        gamma = trial.suggest_float("gamma", 0.001, 1, log=True) if trial else "scale"
        return make_pipeline(
            StandardScaler(),
            SVC(C=c, gamma=gamma, kernel="rbf", probability=True, random_state=random_state),
        )

    if model_name == "rf":
        n_estimators = trial.suggest_int("n_estimators", 50, 500) if trial else 200
        max_depth = trial.suggest_int("max_depth", 3, 20) if trial else 5
        return make_pipeline(
            StandardScaler(),
            RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state),
        )

    if model_name == "knn":
        n_neighbors = trial.suggest_int("n_neighbors", 3, 11) if trial else 5
        return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=n_neighbors, weights="uniform"))

    if model_name == "mlp":
        hidden_layer_sizes = trial.suggest_categorical("hidden_layer_sizes", [(50,), (100,), (50, 50)]) if trial else (50,)
        alpha = trial.suggest_float("alpha", 0.0001, 0.1, log=True) if trial else 0.0001
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                alpha=alpha,
                solver="lbfgs",
                max_iter=2000,
                random_state=random_state,
            ),
        )

    if model_name == "baseline":
        return make_pipeline(
            FunctionTransformer(_select_first_column, validate=False),
            StandardScaler(),
            LogisticRegression(random_state=random_state),
        )

    choices = ", ".join(available_models())
    raise ValueError(f"Unknown model '{model_name}'. Available models: {choices}")


def build_model_from_params(model_name: str, params: dict[str, Any], random_state: int = RANDOM_SEED):
    if model_name == "svm":
        return make_pipeline(
            StandardScaler(),
            SVC(**params, kernel="rbf", probability=True, random_state=random_state),
        )

    if model_name == "rf":
        return make_pipeline(
            StandardScaler(),
            RandomForestClassifier(**params, random_state=random_state),
        )

    if model_name == "knn":
        return make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(**params, weights="uniform"),
        )

    if model_name == "mlp":
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(**params, solver="lbfgs", max_iter=2000, random_state=random_state),
        )

    if model_name == "baseline":
        return build_trial_model(model_name, None, random_state=random_state)

    choices = ", ".join(available_models())
    raise ValueError(f"Unknown model '{model_name}'. Available models: {choices}")


def available_models() -> list[str]:
    return ["baseline", "svm", "rf", "knn", "mlp"]


def select_youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    scores = tpr - fpr
    return float(thresholds[int(np.argmax(scores))])


def _specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(tn / (tn + fp)) if (tn + fp) else 0.0


def _fold_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    auc = float("nan")
    if len(np.unique(y_true)) == 2:
        auc = float(roc_auc_score(y_true, y_prob))
    return {
        "auc": auc,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": _specificity(y_true, y_pred),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }


def tune_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_name: str,
    config: EvaluationConfig,
):
    if model_name == "baseline" or not config.tune:
        return build_trial_model(model_name, None, random_state=config.random_state), {}

    try:
        import optuna
    except ImportError as exc:
        raise ImportError("Optuna is required for hyperparameter tuning. Install requirements or use --fixed-parameters.") from exc

    sampler = optuna.samplers.TPESampler(seed=config.random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial: Any) -> float:
        clf = build_trial_model(model_name, trial, random_state=config.random_state)
        inner_cv = StratifiedKFold(
            n_splits=config.inner_splits,
            shuffle=True,
            random_state=config.random_state,
        )
        scores = []
        for train_index, valid_index in inner_cv.split(X_train, y_train):
            clf.fit(X_train[train_index], y_train[train_index])
            y_valid_prob = clf.predict_proba(X_train[valid_index])[:, 1]
            if len(np.unique(y_train[valid_index])) == 2:
                scores.append(roc_auc_score(y_train[valid_index], y_valid_prob))
        return float(np.mean(scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=config.n_trials)
    return build_model_from_params(model_name, study.best_params, random_state=config.random_state), study.best_params


def evaluate_models(
    X: np.ndarray,
    y: np.ndarray,
    sample_ids: np.ndarray,
    model_names: list[str],
    config: EvaluationConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    class_counts = np.bincount(y.astype(int), minlength=2)
    if np.min(class_counts) < config.n_splits:
        raise ValueError("Each class must contain at least n_splits samples")
    if np.min(class_counts) < config.inner_splits:
        raise ValueError("Each class must contain at least inner_splits samples")

    outer_cv = RepeatedStratifiedKFold(
        n_splits=config.n_splits,
        n_repeats=config.n_repeats,
        random_state=config.random_state,
    )

    rows: list[dict[str, object]] = []

    for fold_id, (train_index, test_index) in enumerate(outer_cv.split(X, y), start=1):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        for model_name in model_names:
            model, best_params = tune_model(X_train, y_train, model_name, config)
            fitted_model = clone(model)
            fitted_model.fit(X_train, y_train)

            y_train_prob = fitted_model.predict_proba(X_train)[:, 1]
            y_test_prob = fitted_model.predict_proba(X_test)[:, 1]
            threshold = select_youden_threshold(y_train, y_train_prob)
            metrics = _fold_metrics(y_test, y_test_prob, threshold)

            row = {
                "model": model_name,
                "fold": fold_id,
                "threshold": threshold,
                "n_train": len(train_index),
                "n_test": len(test_index),
                "test_sample_ids": ";".join(sample_ids[test_index]),
            }
            if best_params:
                row["best_params"] = json.dumps(best_params, sort_keys=True)
            row.update(metrics)
            rows.append(row)

    per_fold = pd.DataFrame(rows)
    summary = (
        per_fold.groupby("model")[["auc", "f1", "accuracy", "sensitivity", "specificity", "precision", "mcc"]]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.columns = ["_".join(column).strip("_") for column in summary.columns.to_flat_index()]
    return per_fold, summary
