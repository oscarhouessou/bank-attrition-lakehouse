"""Simulateur du système d'information d'une banque de détail (type banque de réseau).

Tables produites (comme des extractions du SI, pseudonymisées — aucun nom, IBAN ni adresse) :
  - clients      : profil et équipement (offre, produits détenus, agence/région, âge)
  - operations   : opérations sur compte courant (salaire, carte, prélèvement, virement, frais)
  - interactions : connexions appli, contacts conseiller, réclamations (+ satisfaction)
  - clotures     : clôtures de la relation (source du label d'attrition)

Vérité terrain : la probabilité de partir dépend de l'équipement (un crédit immobilier
retient fortement), des frais/incidents, des réclamations et de la satisfaction.
Signaux précurseurs dans les 90 jours avant la clôture : le salaire n'est plus domicilié,
les virements vers une autre banque augmentent, l'usage de l'appli baisse.

Dérive injectée à partir de `drift_from` (scénario « hausse des frais de tenue de compte ») :
plus de frais et de réclamations, satisfaction en baisse.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SEGMENTS = np.array(["jeune_actif", "particulier", "premium", "senior"])
REGIONS = np.array(["IDF", "Nord", "Est", "Ouest", "Sud-Est", "Sud-Ouest"])
INTERACTION_TYPES = np.array(["connexion_appli", "contact_conseiller", "reclamation"])
COMPLAINT_REASONS = np.array(["frais", "carte_bloquee", "delai_virement", "conseiller_injoignable", "autre"])
CLOSURE_REASONS = np.array(["concurrence_banque_en_ligne", "frais", "demenagement", "insatisfaction", "autre"])


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def generate(n_clients: int = 3000, start: str = "2024-07-01", months: int = 26,
             seed: int = 42, drift_from: str | None = None,
             quality_issues: bool = True) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    start_ts = pd.Timestamp(start)
    end_ts = start_ts + pd.DateOffset(months=months)
    drift_ts = pd.Timestamp(drift_from) if drift_from else None
    n = n_clients

    # ------------------------------------------------------------------ clients
    client_id = np.arange(100001, 100001 + n)
    entree = start_ts + pd.to_timedelta(rng.integers(-3650, int((end_ts - start_ts).days * 0.5), n), "D")
    segment = rng.choice(SEGMENTS, n, p=[0.25, 0.45, 0.12, 0.18])
    age = np.select(
        [segment == "jeune_actif", segment == "senior"],
        [rng.integers(20, 32, n), rng.integers(62, 85, n)],
        rng.integers(30, 62, n))
    region = rng.choice(REGIONS, n, p=[0.30, 0.12, 0.12, 0.16, 0.16, 0.14])
    has_credit_immo = rng.random(n) < np.select([segment == "jeune_actif", segment == "senior"], [0.08, 0.10],
                                                 np.where(segment == "premium", 0.45, 0.30))
    has_assurance_vie = rng.random(n) < np.select([segment == "premium", segment == "senior"], [0.70, 0.55], 0.20)
    has_livret = rng.random(n) < 0.65
    nb_produits = 1 + has_credit_immo + has_assurance_vie + has_livret + rng.poisson(0.8, n)
    salarie = rng.random(n) < np.where(segment == "senior", 0.15, 0.88)   # seniors : pension
    pay_day = rng.integers(0, 28, n)        # jour de versement du salaire / de la pension, propre au client
    revenu = np.round(np.select([segment == "premium", segment == "jeune_actif"], [5200, 1900], 2600)
                      * rng.lognormal(0, 0.25, n), -1)

    card_rate = np.select([segment == "jeune_actif", segment == "premium"], [7, 6], 5) * rng.lognormal(0, 0.35, n)
    app_rate = np.select([segment == "jeune_actif", segment == "senior"], [8, 1.5], 4) * rng.lognormal(0, 0.5, n)
    fee_rate = rng.gamma(1.5, 0.25, n) * np.where(segment == "premium", 0.5, 1.0)     # frais d'incident / mois
    complaint_rate = rng.gamma(1.2, 0.05, n) + 0.08 * fee_rate
    sat_mean = np.clip(rng.normal(3.9, 0.6, n) - 0.9 * (fee_rate > 0.6), 1, 5)

    # Hazard mensuel d'attrition (vérité terrain, inconnue du modèle)
    logit = (-5.0 + 0.9 * (segment == "jeune_actif") - 0.4 * (segment == "senior")
             - 1.6 * has_credit_immo - 0.5 * has_assurance_vie - 0.25 * (nb_produits - 2)
             + 0.9 * fee_rate + 4.0 * complaint_rate + 0.7 * (3.6 - sat_mean)
             + 0.3 * (region == "IDF"))
    hazard = _sigmoid(logit)
    months_to_close = rng.geometric(hazard)
    # l'horloge démarre à l'entrée en relation, ou au début de l'historique pour les clients plus anciens
    clock_start = entree.where(entree > start_ts, start_ts)
    cloture = clock_start + pd.to_timedelta(months_to_close * 30 + rng.integers(0, 30, n), "D")
    closes = cloture < end_ts
    # 70 % des partants transfèrent leur domiciliation de salaire ~2 mois avant
    transfers_salary = closes & (rng.random(n) < 0.7)

    clients = pd.DataFrame({
        "client_id": client_id,
        "date_entree_relation": entree.date,
        "segment": segment,
        "age": age,
        "region": region,
        "nb_produits": nb_produits,
        "has_credit_immo": has_credit_immo.astype(int),
        "has_assurance_vie": has_assurance_vie.astype(int),
        "has_livret": has_livret.astype(int),
        "revenu_mensuel_declare": revenu,
    })
    clotures = pd.DataFrame({
        "client_id": client_id[closes],
        "date_cloture": cloture[closes].date,
        "motif_cloture": rng.choice(CLOSURE_REASONS, closes.sum(), p=[0.35, 0.25, 0.15, 0.15, 0.10]),
    })

    # ------------------------------------------------- opérations & interactions
    months_grid = pd.date_range(start_ts, end_ts - pd.DateOffset(days=1), freq="MS")
    op_parts, it_parts = [], []
    op_n = it_n = 0
    for m_start in months_grid:
        m_end = m_start + pd.DateOffset(months=1)
        a_start = np.maximum(entree.values, m_start.to_datetime64())
        a_end = np.minimum(cloture.values, m_end.to_datetime64())
        active_days = (a_end - a_start) / np.timedelta64(1, "D")
        active = active_days > 0
        frac = np.where(active, active_days / (m_end - m_start).days, 0.0)
        days_to_close = (cloture.values - m_start.to_datetime64()) / np.timedelta64(1, "D")
        pre = active & (days_to_close > 0) & (days_to_close < 90)          # 3 derniers mois
        pre2 = active & (days_to_close > 0) & (days_to_close < 60)         # 2 derniers mois
        drifting = drift_ts is not None and m_start >= drift_ts

        salary_paid = active & (frac > 0.5) & salarie & ~(pre2 & transfers_salary)
        pension_paid = active & (frac > 0.5) & ~salarie
        counts = {
            "salaire": (salary_paid | pension_paid).astype(int),
            "paiement_carte": rng.poisson(card_rate * frac * np.where(pre, 0.55, 1.0)),
            "prelevement": rng.poisson(3.0 * frac * np.where(pre2, 0.5, 1.0)),
            "virement_sortant_autre_banque": rng.poisson(frac * np.where(pre, 1.6, 0.12)),
            "virement_entrant": rng.poisson(0.6 * frac),
            "frais_incident": rng.poisson(fee_rate * frac * (1.8 if drifting else 1.0)),
        }
        for op_type, cnt in counts.items():
            idx = np.repeat(np.arange(n), cnt)
            if idx.size == 0:
                continue
            span = np.maximum(active_days[idx], 1.0)
            if op_type == "salaire":
                ts = np.clip(m_start.to_datetime64() + pay_day[idx].astype("timedelta64[D]")
                             + rng.integers(0, 3, idx.size).astype("timedelta64[D]")
                             + np.timedelta64(9, "h"),
                             a_start[idx], a_end[idx] - np.timedelta64(1, "h"))
            else:
                ts = a_start[idx] + (rng.random(idx.size) * span * 86400).astype("timedelta64[s]")
            amount = {
                "salaire": revenu[idx] * rng.normal(1, 0.03, idx.size),
                "paiement_carte": -rng.lognormal(3.3, 0.8, idx.size),
                "prelevement": -rng.lognormal(4.2, 0.7, idx.size),
                "virement_sortant_autre_banque": -rng.lognormal(6.3, 0.9, idx.size)
                                                  * np.where(pre[idx], 2.5, 1.0),
                "virement_entrant": rng.lognormal(5.0, 1.0, idx.size),
                "frais_incident": -rng.choice([8.0, 12.5, 20.0, 30.0], idx.size)
                                  * (1.3 if drifting else 1.0),
            }[op_type]
            op_parts.append(pd.DataFrame({
                "operation_id": np.char.add("OP", np.char.zfill(np.arange(op_n, op_n + idx.size).astype(str), 9)),
                "client_id": client_id[idx],
                "date_operation": ts,
                "type_operation": op_type,
                "montant": np.round(amount, 2),
            }))
            op_n += idx.size

        it_counts = {
            "connexion_appli": rng.poisson(app_rate * frac * np.where(pre, 0.4, 1.0)),
            "contact_conseiller": rng.poisson(0.3 * frac * np.where(pre, 1.8, 1.0)),
            "reclamation": rng.poisson(complaint_rate * frac * np.where(pre, 2.5, 1.0) * (2.5 if drifting else 1.0)),
        }
        for it_type, cnt in it_counts.items():
            idx = np.repeat(np.arange(n), cnt)
            if idx.size == 0:
                continue
            span = np.maximum(active_days[idx], 1.0)
            ts = a_start[idx] + (rng.random(idx.size) * span * 86400).astype("timedelta64[s]")
            is_c = it_type == "reclamation"
            sat = np.clip(np.round(rng.normal(sat_mean[idx] - (1.0 if drifting else 0.0)
                                              - np.where(pre[idx], 0.6, 0.0), 0.9)), 1, 5)
            it_parts.append(pd.DataFrame({
                "interaction_id": np.char.add("IT", np.char.zfill(np.arange(it_n, it_n + idx.size).astype(str), 8)),
                "client_id": client_id[idx],
                "date_interaction": ts,
                "type_interaction": it_type,
                "motif": np.where(is_c, rng.choice(COMPLAINT_REASONS, idx.size,
                                                   p=[0.45, 0.15, 0.15, 0.15, 0.10] if drifting
                                                   else [0.25, 0.2, 0.2, 0.2, 0.15]), None),
                "satisfaction": np.where(is_c | (it_type == "contact_conseiller"), sat, np.nan),
            }))
            it_n += idx.size

    operations = pd.concat(op_parts, ignore_index=True)
    interactions = pd.concat(it_parts, ignore_index=True)

    if quality_issues:
        # rejeux de fichiers (doublons), montants manquants, type d'opération inconnu
        dup = operations.sample(frac=0.01, random_state=seed)
        operations = pd.concat([operations, dup], ignore_index=True)
        bad = operations.sample(frac=0.004, random_state=seed + 1).index
        operations.loc[bad[: len(bad) // 2], "montant"] = np.nan
        operations.loc[bad[len(bad) // 2:], "type_operation"] = "CODE_INCONNU"

    operations = operations.sort_values("date_operation").reset_index(drop=True)
    interactions = interactions.sort_values("date_interaction").reset_index(drop=True)
    interactions["satisfaction"] = interactions["satisfaction"].astype("Int64")
    return {"clients": clients, "operations": operations,
            "interactions": interactions, "clotures": clotures}


def split_by_date(data: dict[str, pd.DataFrame], cutoff: str) -> tuple[dict, dict]:
    """Historique (< cutoff) et batch incrémental (>= cutoff), comme deux livraisons du SI."""
    c = pd.Timestamp(cutoff)
    cols = {"clients": "date_entree_relation", "operations": "date_operation",
            "interactions": "date_interaction", "clotures": "date_cloture"}
    hist, inc = {}, {}
    for name, col in cols.items():
        d = pd.to_datetime(data[name][col])
        hist[name] = data[name][d < c]
        inc[name] = data[name][d >= c]
    return hist, inc
