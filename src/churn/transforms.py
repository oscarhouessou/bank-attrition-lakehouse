"""Transformations Spark du cas attrition bancaire (silver). Fonctions testables."""
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

import datetime as dt
from churn.config import KEYS, LABEL


OPERATION_TYPES = ["salaire", "paiement_carte", "prelevement", "virement_sortant_autre_banque",
                   "virement_entrant", "frais_incident"]
INTERACTION_TYPES = ["connexion_appli", "contact_conseiller", "reclamation"]


def operation_rules() -> dict:
    """Règles de qualité : nom de la règle -> condition qui doit être VRAIE."""
    return {
        # En Spark, NaN n'est pas NULL et NaN > 0 est VRAI : on teste isnan explicitement
        "montant_renseigne": F.col("montant").isNotNull() & ~F.isnan("montant"),
        "type_operation_connu": F.col("type_operation").isin(OPERATION_TYPES),
        "client_id_renseigne": F.col("client_id").isNotNull(),
        "date_renseignee": F.col("date_operation").isNotNull()
       
    }


def apply_quality_rules(df: DataFrame, rules: dict) -> tuple[DataFrame, DataFrame]:
    """Sépare les lignes valides des lignes rejetées (avec la liste des règles violées)."""
    failed = F.array_compact(F.array(*[
        F.when(~F.coalesce(cond, F.lit(False)), F.lit(name)) for name, cond in rules.items()
    ]))
    tagged = df.withColumn("_failed_rules", failed)
    valid = tagged.where(F.size("_failed_rules") == 0).drop("_failed_rules")
    rejected = tagged.where(F.size("_failed_rules") > 0)
    return valid, rejected


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
# GOLD : population et labels
# ----------------------------------------------------------------------------
def population(clients: DataFrame, clotures: DataFrame, snapshots: DataFrame) -> DataFrame:
    """Clients en relation à chaque date d'observation : entrés avant, pas encore clôturés."""
    snaps = snapshots.select(F.col("snapshot_date").cast("date"))
    return (clients.select("client_id", "date_entree_relation")
            .crossJoin(F.broadcast(snaps))                         # chaque client x chaque date
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