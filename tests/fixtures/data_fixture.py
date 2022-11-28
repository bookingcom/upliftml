from typing import Generator

import h2o  # type: ignore
import pyspark.sql
import pytest
import pandas as pd
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


@pytest.fixture(scope="session")
def df_pd_feature_importance_scores(spark: SparkSession) -> Generator[pd.DataFrame, None, None]:
    feature_names = ["feature_1", "feature_2", "feature_3", "feature_4", "feature_5"]
    importance_scores = [0.9, 0.8, 0.7, 0.6, 0.5]
    df = pd.DataFrame(data={"feature": feature_names, "scores": importance_scores})
    yield df


@pytest.fixture(scope="session")
def df_pd_feature_importance_scores_over_time(spark: SparkSession) -> Generator[pd.DataFrame, None, None]:
    feature_names = ["feature_1", "feature_2", "feature_3", "feature_4", "feature_5"]
    importance_scores = [0.9, 0.8, 0.7, 0.6, 0.5]
    df = pd.DataFrame(
        data={
            "feature": feature_names,
            "2022-05-01": importance_scores,
            "2022-05-02": importance_scores,
            "2022-05-03": importance_scores,
        }
    )
    yield df
