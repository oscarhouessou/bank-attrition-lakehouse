"""Tests de NON-RÉGRESSION du modèle : sur un jeu de référence figé (seed fixe), le pipeline
d'entraînement doit rester au-dessus d'un plancher de performance. Si un refactoring casse
le signal, la CI échoue AVANT le déploiement."""
import datetime as dt

import pandas as pd
import pytest

from churn import data_gen
from churn import modeling as M
from churn import transforms as T
from churn.config import FEATURES, KEYS, month_starts

AUC_FLOOR = 0.75          # plancher de référence (mesuré ≈ 0,85 sur ce jeu)


@pytest.fixture(scope="module")
def dataset(spark):
    d = data_gen.generate(n_clients=2000, start="2024-07-01", months=20, seed=7)
    s = lambda name, col: spark.createDataFrame(d[name].astype({col: str}).astype(str))   # noqa: E731
    ops, _ = T.clean_operations(s("operations", "date_operation"))
    it = T.clean_interactions(s("interactions", "date_interaction"))
    cl = T.clean_clients(s("clients", "date_entree_relation"))
    clo = T.clean_clotures(s("clotures", "date_cloture"))
    snaps = spark.createDataFrame(pd.DataFrame({"snapshot_date": month_starts("2025-04-01", "2025-12-01")}))
    feats = T.build_features(ops, it, cl, clo, snaps)
    return feats.join(T.build_labels(clo, feats, 90), KEYS).toPandas()


def test_reference_dataset_is_reproducible():
    a = data_gen.generate(n_clients=200, months=6, seed=3)["operations"]
    b = data_gen.generate(n_clients=200, months=6, seed=3)["operations"]
    pd.testing.assert_frame_equal(a, b)


@pytest.mark.parametrize("model_type", ["logreg", "hgb"])
def test_model_beats_auc_floor(dataset, model_type):
    cut = dt.date(2025, 10, 1)
    tr, va = dataset[dataset.snapshot_date < cut], dataset[dataset.snapshot_date >= cut]
    pipe = M.build_churn_pipeline(model_type).fit(tr[FEATURES], tr.churn)
    metrics = M.evaluate(va.churn, pipe.predict_proba(va[FEATURES])[:, 1])
    assert metrics["roc_auc"] >= AUC_FLOOR, metrics
    assert metrics["lift_at_10pct"] > 3.0, metrics


def test_pyfunc_wrapper_returns_probabilities(dataset):
    pipe = M.build_churn_pipeline("logreg").fit(dataset[FEATURES], dataset.churn)
    proba = M.ChurnProbaModel(pipe).predict(None, dataset.head(50))
    assert proba.shape == (50,) and ((proba >= 0) & (proba <= 1)).all()


def test_segmentation_names_are_unique(dataset):
    seg = M.SegmentationModel(k=4).fit(dataset)
    assert len(set(seg.names.values())) == 4
    assert seg.predict(None, dataset.head(20)).isin(seg.names.values()).all()
