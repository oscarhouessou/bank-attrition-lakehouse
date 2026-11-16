import numpy as np
import pandas as pd
import pytest

from churn import monitoring as MON
from churn import prescriptive as P


@pytest.fixture
def scored():
    rng = np.random.default_rng(0)
    n = 300
    return pd.DataFrame({"client_id": np.arange(n),
                         "p_churn": rng.uniform(0, 1, n),
                         "clv": rng.lognormal(5, 0.8, n)})


@pytest.mark.parametrize("method", ["milp", "greedy"])
def test_campaign_respects_constraints(scored, method):
    plan = P.optimize_campaign(scored, budget=500, method=method)
    assert plan.cost.sum() <= 500 + 1e-6
    assert plan.client_id.is_unique            # au plus une action par client
    assert (plan.net_gain > 0).all()             # jamais d'action à perte


def test_milp_is_at_least_as_good_as_greedy(scored):
    milp = P.optimize_campaign(scored, budget=500, method="milp").net_gain.sum()
    greedy = P.optimize_campaign(scored, budget=500, method="greedy").net_gain.sum()
    assert milp >= greedy - 1e-6


def test_no_action_when_nobody_at_risk():
    s = pd.DataFrame({"client_id": [1, 2], "p_churn": [0.0, 0.0], "clv": [100.0, 200.0]})
    assert P.optimize_campaign(s, budget=1000).empty


def test_psi_identical_is_zero_and_shift_is_detected():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, 5000)
    assert MON.psi(a, a) == pytest.approx(0.0, abs=1e-9)
    assert MON.psi(a, rng.normal(0, 1, 5000)) < 0.05
    assert MON.psi(a, rng.normal(1.0, 1, 5000)) > 0.25


def test_psi_handles_constant_and_nan():
    assert np.isfinite(MON.psi(np.ones(100), np.ones(100)))
    assert np.isnan(MON.psi(np.array([np.nan]), np.array([1.0])))


def test_status_and_overall():
    assert MON.status(0.05, 0.1, 0.25) == "OK"
    assert MON.status(0.15, 0.1, 0.25) == "WARN"
    assert MON.status(0.30, 0.1, 0.25) == "ALERT"
    rep = pd.DataFrame({"status": ["OK", "WARN"]})
    assert MON.overall_status(rep) == "WARN"


def test_customer_value_rewards_equipment():
    df = pd.DataFrame({"nb_produits": [1, 4], "has_credit_immo": [0, 1], "has_assurance_vie": [0, 1]})
    v = P.customer_value(df)
    assert v.iloc[1] > v.iloc[0] > 0
