from typing import Generator

import h2o  # type: ignore
import pyspark.sql
import pytest
from pyspark.sql import SparkSession

from upliftml.datasets import simulate_randomized_trial


@pytest.fixture(scope="session")
def df_spark_cont(spark: SparkSession) -> Generator[pyspark.sql.DataFrame, None, None]:
    df_pd = simulate_randomized_trial(n=1000, p=6, sigma=1.0)
    df_spark = spark.createDataFrame(df_pd)
    yield df_spark


@pytest.fixture(scope="session")
def df_spark_binary(spark: SparkSession) -> Generator[pyspark.sql.DataFrame, None, None]:
    df_pd = simulate_randomized_trial(n=1000, p=6, sigma=1.0, binary_outcome=True)
    df_spark = spark.createDataFrame(df_pd)
    yield df_spark


@pytest.fixture(scope="session")
def df_h2o_cont() -> Generator[h2o.H2OFrame, None, None]:
    df_pd = simulate_randomized_trial(n=1000, p=6, sigma=1.0)
    df_h2o = h2o.H2OFrame(df_pd)
    yield df_h2o


@pytest.fixture(scope="session")
def df_h2o_binary() -> Generator[h2o.H2OFrame, None, None]:
    df_pd = simulate_randomized_trial(n=5000, p=6, sigma=1.0, binary_outcome=True)
    df_h2o = h2o.H2OFrame(df_pd)
    yield df_h2o


@pytest.fixture(scope="session")
def df_h2o_binary_val() -> Generator[h2o.H2OFrame, None, None]:
    df_pd = simulate_randomized_trial(n=1000, p=6, sigma=1.0, binary_outcome=True)
    df_h2o = h2o.H2OFrame(df_pd)
    yield df_h2o


@pytest.fixture(scope="session")
def df_spark_binary_with_cost(spark: SparkSession) -> Generator[h2o.H2OFrame, None, None]:
    df_pd = simulate_randomized_trial(n=1000, p=6, sigma=1.0, binary_outcome=True, add_cost_benefit=True)
    df_spark = spark.createDataFrame(df_pd)
    yield df_spark
