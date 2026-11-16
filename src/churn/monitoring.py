"""Monitoring des modèles en production : dérive des données (PSI), dérive des prédictions,
performance réelle quand les labels arrivent (90 jours plus tard)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10, eps: float = 1e-4) -> float:
    """Population Stability Index. < 0,10 stable ; 0,10 à 0,25 à surveiller ; > 0,25 dérive."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected, actual = expected[~np.isnan(expected)], actual[~np.isnan(actual)]
    if expected.size == 0 or actual.size == 0:
        return float("nan")
    # 1) on découpe la distribution de référence en 10 tranches de même effectif (déciles)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if edges.size < 3:                                  # variable quasi constante
        edges = np.unique(np.concatenate([edges, [edges[0] - 1, edges[-1] + 1]]))
    edges[0], edges[-1] = -np.inf, np.inf
    # 2) part de la population de référence et de la population actuelle dans chaque tranche
    e = np.histogram(expected, edges)[0] / expected.size
    a = np.histogram(actual, edges)[0] / actual.size
    e, a = np.clip(e, eps, None), np.clip(a, eps, None)
    # 3) on somme les écarts, pondérés par le log du rapport
    return float(np.sum((a - e) * np.log(a / e)))


def status(value: float, warn: float, alert: float) -> str:
    if value is None or np.isnan(value):
        return "UNKNOWN"
    return "ALERT" if value >= alert else "WARN" if value >= warn else "OK"


def drift_report(baseline: pd.DataFrame, current: pd.DataFrame, features: list[str],
                 warn: float = 0.10, alert: float = 0.25) -> pd.DataFrame:
    """Un PSI et un statut par feature."""
    rows = []
    for f in features:
        v = psi(baseline[f].to_numpy(), current[f].to_numpy())
        rows.append({"metric": f"psi_{f}", "feature": f, "value": v, "status": status(v, warn, alert)})
    return pd.DataFrame(rows)


def performance_check(y_true, y_score, min_auc: float) -> dict:
    """AUC réelle, une fois les labels connus."""
    from sklearn.metrics import roc_auc_score
    auc = float(roc_auc_score(y_true, y_score))
    return {"metric": "realized_roc_auc", "feature": None, "value": auc,
            "status": "ALERT" if auc < min_auc else "OK"}


def overall_status(report: pd.DataFrame) -> str:
    s = set(report["status"])
    return "ALERT" if "ALERT" in s else "WARN" if "WARN" in s else "OK"