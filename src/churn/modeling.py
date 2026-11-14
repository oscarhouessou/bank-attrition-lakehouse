"""Modèles (attrition + segmentation) et métriques. Indépendant de Spark : pandas / scikit-learn."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, SEGMENT_FEATURES

try:  # mlflow n'est pas requis pour les tests unitaires
    import mlflow.pyfunc
    _PythonModel = mlflow.pyfunc.PythonModel
except Exception:  # pragma: no cover
    _PythonModel = object


# ----------------------------------------------------------------------------
# Attrition (utilisé à l'étape 10)
# ----------------------------------------------------------------------------
def build_churn_pipeline(model_type: str = "hgb", **params) -> Pipeline:
    if model_type == "logreg":   # baseline interprétable
        pre = ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                              ("sc", StandardScaler())]), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ])
        clf = LogisticRegression(max_iter=1000, C=params.get("C", 1.0))
    else:                         # gradient boosting, gère nativement les valeurs manquantes
        pre = ColumnTransformer([
            ("num", "passthrough", NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ])
        clf = HistGradientBoostingClassifier(
            learning_rate=params.get("learning_rate", 0.08),
            max_depth=params.get("max_depth", 6),
            max_leaf_nodes=params.get("max_leaf_nodes", 31),
            min_samples_leaf=params.get("min_samples_leaf", 40),
            l2_regularization=params.get("l2_regularization", 0.0),
            max_iter=params.get("max_iter", 300),
            early_stopping=True, random_state=42)
    return Pipeline([("pre", pre), ("clf", clf)])


def lift_at_k(y_true, y_score, k: float = 0.1) -> float:
    """Taux d'attrition dans le top k % des scores / taux d'attrition moyen."""
    y_true = np.asarray(y_true)
    n = max(1, int(len(y_true) * k))
    top = np.argsort(-np.asarray(y_score))[:n]
    base = y_true.mean()
    return float(y_true[top].mean() / base) if base > 0 else float("nan")


def evaluate(y_true, y_score) -> dict:
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "brier": float(brier_score_loss(y_true, y_score)),
        "lift_at_10pct": lift_at_k(y_true, y_score, 0.10),
        "churn_rate": float(np.mean(y_true)),
    }


class ChurnProbaModel(_PythonModel):
    """Wrapper MLflow : renvoie la PROBABILITÉ d'attrition au lieu de la classe 0/1."""

    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline

    def predict(self, context, model_input: pd.DataFrame, params=None):
        X = model_input[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        return self.pipeline.predict_proba(X)[:, 1]


# ----------------------------------------------------------------------------
# Segmentation (cette étape)
# ----------------------------------------------------------------------------
def _segment_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Prépare les variables : log pour écraser les très grandes valeurs (revenus, dépenses)."""
    X = df[SEGMENT_FEATURES].astype(float).copy()
    for c in ["n_app_logins_30d", "card_spend_90d", "revenu_mensuel_declare", "fees_90d", "n_complaints_90d"]:
        X[c] = np.log1p(X[c].clip(lower=0))
    return X


class SegmentationModel(_PythonModel):
    """Normalisation + K-Means + noms lisibles par le métier, empaquetés en un seul modèle MLflow."""

    def __init__(self, k: int = 4, seed: int = 42):
        self.k, self.seed = k, seed
        self.scaler, self.km, self.names = StandardScaler(), None, {}

    def fit(self, df: pd.DataFrame):
        Z = self.scaler.fit_transform(_segment_matrix(df))
        self.km = KMeans(n_clusters=self.k, n_init=10, random_state=self.seed).fit(Z)
        self.names = name_segments(pd.DataFrame(self.km.cluster_centers_, columns=SEGMENT_FEATURES))
        self.silhouette_ = float(silhouette_score(Z, self.km.labels_, sample_size=min(5000, len(Z)),
                                                  random_state=self.seed))
        return self

    def predict(self, context, model_input: pd.DataFrame, params=None):
        ids = self.km.predict(self.scaler.transform(_segment_matrix(model_input)))
        return pd.Series([self.names[i] for i in ids], index=model_input.index)


def name_segments(centroids: pd.DataFrame) -> dict[int, str]:
    """Nomme chaque cluster d'après son centre (valeurs standardisées : 0 = moyenne)."""
    names = {}
    for i, c in centroids.iterrows():
        if c.n_complaints_90d > 0.8 or c.fees_90d > 0.8:
            n = "Fragilisés / irritants"
        elif c.nb_produits > 0.6 or c.revenu_mensuel_declare > 0.8:
            n = "Patrimoniaux multi-équipés"
        elif c.n_app_logins_30d > 0.4:
            n = "Digitaux actifs"
        else:
            n = "Peu engagés"
        names[i] = n
    seen = {}
    for i in sorted(names):
        seen[names[i]] = seen.get(names[i], 0) + 1
        if seen[names[i]] > 1:
            names[i] = f"{names[i]} {seen[names[i]]}"
    return names


def choose_k(df: pd.DataFrame, ks=(3, 4, 5, 6)) -> tuple[int, dict]:
    """Teste plusieurs k et garde celui qui a le meilleur score de silhouette."""
    scores = {k: SegmentationModel(k).fit(df).silhouette_ for k in ks}
    return max(scores, key=scores.get), scores