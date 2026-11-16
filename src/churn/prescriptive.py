"""Analyse prescriptive : quelle action de rétention pour quel client, sous budget ?

Problème de sac à dos à choix multiples :
  maximiser  la somme des gains des actions choisies
  sous       au plus une action par client
             coût total <= budget
avec gain = P(attrition) x uplift x CLV - coût   (valeur espérée sauvée, nette du coût)

Les uplifts sont des hypothèses métier : en production, on les mesure par test A/B.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_ACTIONS = pd.DataFrame([
    # action,                         coût €, uplift (réduction relative du risque)
    ("message_appli_personnalise",     1.0, 0.04),
    ("geste_commercial_frais_6_mois", 40.0, 0.15),
    ("rdv_conseiller_bilan",          70.0, 0.30),   # coût = temps conseiller valorisé
], columns=["action", "cost", "uplift"])


def customer_value(df: pd.DataFrame, horizon_years: float = 3.0) -> pd.Series:
    """Valeur client simplifiée : PNB annuel estimé d'après l'équipement x horizon de rétention.
    En réel : le PNB observé par client, fourni par le contrôle de gestion."""
    pnb = 90 + 35 * df.nb_produits + 260 * df.has_credit_immo + 140 * df.has_assurance_vie
    return pnb * horizon_years


def expected_gains(scored: pd.DataFrame, actions: pd.DataFrame = DEFAULT_ACTIONS) -> pd.DataFrame:
    """Une ligne par (client, action) avec le gain espéré net."""
    s = scored[["client_id", "p_churn", "clv"]]
    g = s.merge(actions, how="cross")                          # chaque client x chaque action
    g["expected_saved_value"] = g.p_churn * g.uplift * g.clv
    g["net_gain"] = g.expected_saved_value - g.cost
    return g


def optimize_campaign(scored: pd.DataFrame, budget: float,
                      actions: pd.DataFrame = DEFAULT_ACTIONS, method: str = "auto") -> pd.DataFrame:
    """Une action par client retenu. method = 'milp' (exact) ou 'greedy' (heuristique)."""
    g = expected_gains(scored, actions)
    g = g[g.net_gain > 0].reset_index(drop=True)              # jamais d'action à perte
    if g.empty:
        return g.iloc[0:0]
    if method in ("auto", "milp"):
        try:
            return _solve_milp(g, budget)
        except ImportError:
            if method == "milp":
                raise
    return _solve_greedy(g, budget)


def _solve_milp(g: pd.DataFrame, budget: float) -> pd.DataFrame:
    """Solution exacte : programmation linéaire en nombres entiers (scipy)."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import csr_matrix

    n = len(g)
    cust_codes, _ = pd.factorize(g.client_id)
    one_per_customer = csr_matrix((np.ones(n), (cust_codes, np.arange(n))))   # au plus 1 action / client
    budget_row = g.cost.to_numpy()[None, :]                                     # coût total <= budget
    res = milp(
        c=-g.net_gain.to_numpy(),                       # milp minimise : on minimise -gain
        constraints=[LinearConstraint(one_per_customer, 0, 1),
                     LinearConstraint(budget_row, 0, budget)],
        integrality=np.ones(n), bounds=Bounds(0, 1),    # variables 0/1 : action choisie ou non
        options={"time_limit": 60})
    if res.x is None:
        return _solve_greedy(g, budget)
    return g[res.x > 0.5].sort_values("net_gain", ascending=False).reset_index(drop=True)


def _solve_greedy(g: pd.DataFrame, budget: float) -> pd.DataFrame:
    """Heuristique : meilleure action par client, triée par rentabilité, jusqu'au budget."""
    best = g.loc[g.groupby("client_id").net_gain.idxmax()].copy()
    best["roi"] = best.net_gain / best.cost
    best = best.sort_values("roi", ascending=False)
    best = best[best.cost.cumsum() <= budget]
    return best.drop(columns="roi").sort_values("net_gain", ascending=False).reset_index(drop=True)