import datetime as dt

import pandas as pd
import pytest

from churn import transforms as T

D = dt.date
OPS_SCHEMA = "operation_id string, client_id string, date_operation string, type_operation string, montant string"


@pytest.fixture
def ops(spark):
    rows = [
        ("O1", "1", "2026-01-26 09:00:00", "salaire", "2500"),
        ("O2", "1", "2026-02-10 12:00:00", "paiement_carte", "-40.5"),
        ("O2", "1", "2026-02-10 12:00:00", "paiement_carte", "-40.5"),           # rejeu (doublon)
        ("O3", "1", "2026-02-20 12:00:00", "virement_sortant_autre_banque", "-1500"),
        ("O4", "1", "2026-03-05 12:00:00", "paiement_carte", "-20"),             # futur : ignoré
        ("O5", "2", "2026-02-15 12:00:00", "frais_incident", None),              # montant manquant
        ("O6", "2", "2026-02-16 12:00:00", "CODE_INCONNU", "-12"),               # type inconnu
        ("O7", "2", "2026-02-17 12:00:00", "frais_incident", "-12.5"),
        ("O8", "2", "2025-11-25 12:00:00", "salaire", "1900"),              # > 90 j
    ]
    return spark.createDataFrame(rows, OPS_SCHEMA)


@pytest.fixture
def clients(spark):
    return T.clean_clients(spark.createDataFrame(pd.DataFrame({
        "client_id": ["1", "2", "3", "4"],
        "date_entree_relation": ["2020-01-01", "2024-06-01", "2023-01-01", "2026-03-15"],
        "segment": ["particulier", "jeune_actif", "senior", "premium"],
        "age": ["45", "27", "70", "50"], "region": ["IDF", "Sud-Est", "Nord", "IDF"],
        "nb_produits": ["4", "1", "2", "5"], "has_credit_immo": ["1", "0", "0", "1"],
        "has_assurance_vie": ["0", "0", "1", "1"], "has_livret": ["1", "0", "1", "1"],
        "revenu_mensuel_declare": ["2500", "1900", "1500", "6000"]})))


@pytest.fixture
def clotures(spark):
    return T.clean_clotures(spark.createDataFrame(pd.DataFrame({
        "client_id": ["2", "3"], "date_cloture": ["2026-04-20", "2026-01-10"],
        "motif_cloture": ["frais", "demenagement"]})))


@pytest.fixture
def interactions(spark):
    return T.clean_interactions(spark.createDataFrame(pd.DataFrame({
        "interaction_id": ["I1", "I2", "I3", "I4"],
        "client_id": ["2", "2", "1", "1"],
        "date_interaction": ["2026-02-01 10:00:00", "2026-02-20 10:00:00",
                             "2026-02-25 10:00:00", "2026-02-26 10:00:00"],
        "type_interaction": ["reclamation", "reclamation", "connexion_appli", "connexion_appli"],
        "motif": ["frais", "frais", None, None],
        "satisfaction": ["2", "9", None, None]})))      # 9 = invalide -> NULL


@pytest.fixture
def snaps(spark):
    return spark.createDataFrame([(D(2026, 3, 1),)], "snapshot_date date")


def test_quality_rules_and_dedupe(ops):
    valid, rejected = T.clean_operations(ops)
    assert sorted(r.operation_id for r in valid.collect()) == ["O1", "O2", "O3", "O4", "O7", "O8"]
    rej = {r.operation_id: r._failed_rules for r in rejected.collect()}
    assert rej == {"O5": ["montant_renseigne"], "O6": ["type_operation_connu"]}


def test_nan_amount_is_rejected(spark):
    df = spark.createDataFrame([("X", "1", "2026-01-01 00:00:00", "salaire", "NaN")], OPS_SCHEMA)
    valid, rejected = T.clean_operations(df)
    assert valid.count() == 0 and rejected.count() == 1


def test_population_excludes_closed_and_not_yet_clients(clients, clotures, snaps):
    pop = {r.client_id for r in T.population(clients, clotures, snaps).collect()}
    assert pop == {1, 2}    # 3 clôturé avant, 4 entré après la date d'observation


def test_features_point_in_time(ops, interactions, clients, clotures, snaps):
    valid, _ = T.clean_operations(ops)
    f = {r.client_id: r for r in T.build_features(valid, interactions, clients, clotures, snaps).collect()}
    c1 = f[1]
    assert c1.recency_days == (D(2026, 3, 1) - D(2026, 2, 20)).days      # O4 (futur) ignorée
    assert c1.card_spend_90d == pytest.approx(40.5)
    assert c1.ext_transfer_out_90d == pytest.approx(1500.0) and c1.n_ext_transfers_90d == 1
    assert c1.n_salary_90d == 1 and c1.days_since_salary == 34
    assert c1.n_app_logins_30d == 2 and c1.n_complaints_90d == 0
    assert c1.has_credit_immo == 1 and c1.tenure_days > 2000
    c2 = f[2]
    assert c2.fees_90d == 1 and c2.n_complaints_90d == 2
    assert c2.avg_satisfaction_90d == pytest.approx(2.0)                   # la note invalide est ignorée
    assert c2.n_salary_90d == 0 and c2.days_since_salary == 96


def test_labels_attrition_90_days(clients, clotures, snaps):
    pop = T.population(clients, clotures, snaps)
    lab = {r.client_id: r.churn for r in T.build_labels(clotures, pop, 90).collect()}
    assert lab == {1: 0, 2: 1}    # client 2 clôture le 20/04, dans les 90 j


def test_matured_snapshots():
    snaps = [D(2026, 3, 1), D(2026, 4, 1), D(2026, 5, 1)]
    assert T.matured_snapshots(snaps, D(2026, 7, 1), 90) == [D(2026, 3, 1), D(2026, 4, 1)]


def test_cloture_exactement_a_90_jours(spark, clients):
    """Une clôture pile à S + 90 jours appartient à la fenêtre SUIVANTE (intervalle [S, S+90[)."""
    clo = T.clean_clotures(spark.createDataFrame(pd.DataFrame({
        "client_id": ["1"], "date_cloture": ["2026-05-30"],      # 1er mars + 90 jours = 30 mai
        "motif_cloture": ["frais"]})))
    snaps = spark.createDataFrame([(D(2026, 3, 1),)], "snapshot_date date")
    pop = T.population(clients, clo, snaps)
    lab = {r.client_id: r.churn for r in T.build_labels(clo, pop, 90).collect()}
    assert lab[1] == 0     # le 30 mai est EXCLU de la fenêtre [1er mars, 30 mai[


def test_cloture_a_89_jours(spark, clients):
    """Une clôture la veille de S + 90 jours est bien comptée comme attrition."""
    clo = T.clean_clotures(spark.createDataFrame(pd.DataFrame({
        "client_id": ["1"], "date_cloture": ["2026-05-29"],      # 89 jours après le 1er mars
        "motif_cloture": ["frais"]})))
    snaps = spark.createDataFrame([(D(2026, 3, 1),)], "snapshot_date date")
    pop = T.population(clients, clo, snaps)
    lab = {r.client_id: r.churn for r in T.build_labels(clo, pop, 90).collect()}
    assert lab[1] == 1     # le 29 mai est DANS la fenêtre
