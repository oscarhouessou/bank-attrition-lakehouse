import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.fixture(scope="session")
def spark():
    """SparkSession locale : les mêmes transformations que sur Databricks, sans cluster.
    (Alternative : Databricks Connect pour tester contre un vrai workspace.)"""
    from pyspark.sql import SparkSession
    s = (SparkSession.builder.master("local[2]").appName("churn-tests")
         .config("spark.sql.shuffle.partitions", "2")
         .config("spark.ui.enabled", "false")
         .config("spark.sql.session.timeZone", "UTC")
         .getOrCreate())
    s.sparkContext.setLogLevel("ERROR")
    yield s
    s.stop()
