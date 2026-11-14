"""Transformations Spark (silver & gold) du cas « attrition bancaire ».
Fonctions pures DataFrame -> DataFrame, testables en local avec pytest + SparkSession."""
from __future__ import annotations

import datetime as dt

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from churn.config import KEYS, LABEL

OPERATION_TYPES = ["salaire", "paiement_carte", "prelevement", "virement_sortant_autre_banque",
                   "virement_entrant", "frais_incident"]
INTERACTION_TYPES = ["connexion_appli", "contact_conseiller", "reclamation"]


# ----------------------------------------------------------------------------
# SILVER : typage, règles de qualité, dédoublonnage
# ----------------------------------------------------------------------------
def operation_rules() -> dict:
    """Règles de qualité : nom de la règle -> condition qui doit être VRAIE.
    NB : en Spark, NaN n'est pas NULL et NaN > 0 est VRAI -> tester isnan explicitement."""
    return {
        "montant_renseigne": F.col("montant").isNotNull() & ~F.isnan("montant"),
        "type_operation_connu": F.col("type_operation").isin(OPERATION_TYPES),
        "client_id_renseigne": F.col("client_id").isNotNull(),
        "date_renseignee": F.col("date_operation").isNotNull(),
    }


def apply_quality_rules(df: DataFrame, rules: dict) -> tuple[DataFrame, DataFrame]:
    """(lignes valides, lignes rejetées + liste des règles violées)."""
    failed = F.array_compact(F.array(*[
        F.when(~F.coalesce(cond, F.lit(False)), F.lit(name)) for name, cond in rules.items()
    ]))
    tagged = df.withColumn("_failed_rules", failed)
    return (tagged.where(F.size("_failed_rules") == 0).drop("_failed_rules"),
            tagged.where(F.size("_failed_rules") > 0))


def clean_operations(bronze: DataFrame) -> tuple[DataFrame, DataFrame]:
    typed = (bronze
             .select(F.col("operation_id").cast("string"),
                     F.col("client_id").cast("long"),
                     F.to_timestamp("date_operation").alias("date_operation"),
                     F.lower(F.trim("type_operation")).alias("type_operation"),
                     F.col("montant").cast("double"))
             .dropDuplicates(["operation_id"]))
    valid, rejected = apply_quality_rules(typed, operation_rules())
    return valid.withColumn("op_date", F.to_date("date_operation")), rejected


def clean_interactions(bronze: DataFrame) -> DataFrame:
    return (bronze
            .select(F.col("interaction_id").cast("string"),
                    F.col("client_id").cast("long"),
                    F.to_timestamp("date_interaction").alias("date_interaction"),
                    F.col("type_interaction").cast("string"),
                    F.col("motif").cast("string"),
                    F.col("satisfaction").cast("int"))
            .where(F.col("type_interaction").isin(INTERACTION_TYPES))
            .withColumn("satisfaction", F.when(F.col("satisfaction").between(1, 5), F.col("satisfaction")))
            .dropDuplicates(["interaction_id"])
            .withColumn("it_date", F.to_date("date_interaction")))


def clean_clients(bronze: DataFrame) -> DataFrame:
    return (bronze
            .select(F.col("client_id").cast("long"),
                    F.to_date("date_entree_relation").alias("date_entree_relation"),
                    F.col("segment").cast("string"),
                    F.col("age").cast("int"),
                    F.col("region").cast("string"),
                    F.col("nb_produits").cast("int"),
                    F.col("has_credit_immo").cast("int"),
                    F.col("has_assurance_vie").cast("int"),
                    F.col("has_livret").cast("int"),
                    F.col("revenu_mensuel_declare").cast("double"))
            .dropDuplicates(["client_id"]))


def clean_clotures(bronze: DataFrame) -> DataFrame:
    return (bronze
            .select(F.col("client_id").cast("long"),
                    F.to_date("date_cloture").alias("date_cloture"),
                    F.col("motif_cloture").cast("string"))
            .dropDuplicates(["client_id"]))


# ----------------------------------------------------------------------------
# GOLD : population, labels, features point-in-time
# ----------------------------------------------------------------------------
def population(clients: DataFrame, clotures: DataFrame, snapshots: DataFrame) -> DataFrame:
    """Clients en relation à chaque date d'observation : entrés avant, pas encore clôturés."""
    snaps = snapshots.select(F.col("snapshot_date").cast("date"))
    return (clients.select("client_id", "date_entree_relation")
            .crossJoin(F.broadcast(snaps))
            .where(F.col("date_entree_relation") < F.col("snapshot_date"))
            .join(clotures.select("client_id", "date_cloture"), "client_id", "left")
            .where(F.col("date_cloture").isNull() | (F.col("date_cloture") >= F.col("snapshot_date")))
            .select(*KEYS))


def build_labels(clotures: DataFrame, pop: DataFrame, horizon_days: int = 90) -> DataFrame:
    """churn = 1 si la relation est clôturée dans [snapshot_date, snapshot_date + horizon[."""
    return (pop.select(*KEYS).distinct()
            .join(clotures.select("client_id", "date_cloture"), "client_id", "left")
            .withColumn(LABEL, F.when(
                (F.col("date_cloture") >= F.col("snapshot_date")) &
                (F.col("date_cloture") < F.date_add("snapshot_date", horizon_days)),
                1).otherwise(0))
            .select(*KEYS, LABEL))


def matured_snapshots(snapshot_dates: list, data_end_exclusive: dt.date, horizon_days: int) -> list:
    """Dates d'observation dont la fenêtre de label est entièrement observée."""
    return [d for d in snapshot_dates if d + dt.timedelta(days=horizon_days) <= data_end_exclusive]


def _n(cond):
    """Compte les lignes qui vérifient la condition."""
    return F.sum(F.when(cond, 1).otherwise(0))


def _s(cond, col):
    """Somme la colonne sur les lignes qui vérifient la condition."""
    return F.sum(F.when(cond, F.col(col)).otherwise(0.0))


def build_features(ops: DataFrame, interactions: DataFrame, clients: DataFrame,
                   clotures: DataFrame, snapshots: DataFrame) -> DataFrame:
    """Features à chaque date d'observation, calculées UNIQUEMENT avec le passé (< snapshot_date)."""
    pop = population(clients, clotures, snapshots)
    snaps = pop.select("snapshot_date").distinct()

    # --- Opérations des 180 jours précédant chaque date S ---
    o = (ops.select("client_id", "op_date", "type_operation", "montant")
         .crossJoin(F.broadcast(snaps))
         .where((F.col("op_date") < F.col("snapshot_date")) &
                (F.col("op_date") >= F.date_sub("snapshot_date", 180)))
         .withColumn("age_days", F.datediff("snapshot_date", "op_date")))
    t = F.col("type_operation")
    in90 = F.col("age_days") <= 90
    op_feats = o.groupBy(*KEYS).agg(
        F.min("age_days").alias("recency_days"),
        _n(F.col("age_days") <= 30).alias("n_ops_30d"),
        _n(in90).alias("n_ops_90d"),
        (-_s(in90 & (t == "paiement_carte"), "montant")).alias("card_spend_90d"),
        _s(in90 & (F.col("montant") > 0), "montant").alias("inflow_90d"),
        _n(in90 & (t == "salaire")).alias("n_salary_90d"),
        F.min(F.when(t == "salaire", F.col("age_days"))).alias("days_since_salary"),
        (-_s(in90 & (t == "virement_sortant_autre_banque"), "montant")).alias("ext_transfer_out_90d"),
        _n(in90 & (t == "virement_sortant_autre_banque")).alias("n_ext_transfers_90d"),
        _n(in90 & (t == "frais_incident")).alias("fees_90d"))

    # --- Interactions des 90 jours précédant S ---
    i = (interactions.select("client_id", "it_date", "type_interaction", "satisfaction")
         .crossJoin(F.broadcast(snaps))
         .where((F.col("it_date") < F.col("snapshot_date")) &
                (F.col("it_date") >= F.date_sub("snapshot_date", 90)))
         .withColumn("age_days", F.datediff("snapshot_date", "it_date")))
    ti = F.col("type_interaction")
    it_feats = i.groupBy(*KEYS).agg(
        _n((ti == "connexion_appli") & (F.col("age_days") <= 30)).alias("n_app_logins_30d"),
        _n(ti == "connexion_appli").alias("n_app_logins_90d"),
        _n(ti == "reclamation").alias("n_complaints_90d"),
        _n(ti == "contact_conseiller").alias("n_advisor_contacts_90d"),
        F.avg("satisfaction").alias("avg_satisfaction_90d"))

    # --- Assemblage : on part de la population pour garder les clients sans activité ---
    zero_cols = ["n_ops_30d", "n_ops_90d", "card_spend_90d", "inflow_90d", "n_salary_90d",
                 "ext_transfer_out_90d", "n_ext_transfers_90d", "fees_90d", "n_app_logins_30d",
                 "n_app_logins_90d", "n_complaints_90d", "n_advisor_contacts_90d"]
    feats = (pop
             .join(op_feats, KEYS, "left")
             .join(it_feats, KEYS, "left")
             .fillna(0, subset=zero_cols)
             .fillna(181, subset=["recency_days", "days_since_salary"])   # 181 = « aucun en 180 j »
             .withColumn("app_trend_30_90",
                         F.when(F.col("n_app_logins_90d") > 0,
                                F.col("n_app_logins_30d") / (F.col("n_app_logins_90d") / 3.0))
                         .otherwise(0.0))
             .drop("n_app_logins_90d")
             .join(clients, "client_id")
             .withColumn("tenure_days", F.datediff("snapshot_date", "date_entree_relation"))
             .drop("date_entree_relation"))
    # avg_satisfaction_90d reste NULL sans interaction notée : le modèle gère les valeurs manquantes
    return feats.select(*KEYS, *[c for c in feats.columns if c not in KEYS])