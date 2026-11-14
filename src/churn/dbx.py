"""Helpers spécifiques Databricks (utilisés seulement dans les notebooks)."""
from __future__ import annotations

from churn.config import Config


def get_config(dbutils) -> Config:
    """Lit catalog/schema depuis les widgets (alimentés par les paramètres du job du bundle)."""
    for name, default in (("catalog", "workspace"), ("schema", "attrition_dev")):
        try:
            dbutils.widgets.get(name)
        except Exception:
            dbutils.widgets.text(name, default)
    return Config(catalog=dbutils.widgets.get("catalog"), schema=dbutils.widgets.get("schema"))


def write_delta(spark, pdf, table: str, mode: str = "append"):
    """Écrit un DataFrame pandas dans une table Delta Unity Catalog."""
    (spark.createDataFrame(pdf).write.mode(mode)
     .option("mergeSchema", "true").saveAsTable(table))


def table_version(spark, table: str) -> int:
    """Version Delta courante : loggée dans MLflow pour la reproductibilité (time travel)."""
    return spark.sql(f"DESCRIBE HISTORY {table} LIMIT 1").first()["version"]
