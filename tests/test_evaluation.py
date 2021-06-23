# type: ignore
import numpy as np
import pandas as pd
import pyspark.sql
import pyspark.sql.functions as F
import pytest
from pyspark.sql import SparkSession

from upliftml.evaluation import (
    compute_qini_coefficient,
    estimate_ate,
    estimate_cate_lift,
    estimate_cate_per_bucket,
    estimate_cate_per_quantile,
    estimate_cum_iroi,
    estimate_iroi,
    estimate_iroi_per_bucket,
    estimate_iroi_per_quantile,
    estimate_qini,
    estimate_roi,
    estimate_target_rate_per_bucket,
    estimate_target_rate_per_quantile,
)


def test_compute_ate(spark: SparkSession, df_spark_cont: pyspark.sql.DataFrame) -> None:
    actuals = (
        df_spark_cont.groupby("treatment").agg(F.mean("outcome").alias("target_rate")).toPandas().set_index("treatment")
    )

    # Test w/o bootstrapping #
    results = estimate_ate(df_spark_cont, bootstrap=False)
    outcome_rate_control = results["target_rate_control"]
    outcome_rate_treated = results["target_rate_treated"]
    ate = results["ate"]

    assert outcome_rate_control == pytest.approx(actuals.loc[0])
    assert outcome_rate_treated == pytest.approx(actuals.loc[1])
    assert ate == pytest.approx(actuals.loc[1] - actuals.loc[0])

    # Test with bootstrapping #
    results = estimate_ate(df_spark_cont, bootstrap=True, n_bootstraps=100, ci_quantiles=[0.1, 0.9])

    assert len(results) == 9

    # check that lower bound is not higher than point estimate
    assert results["target_rate_control_lower"] <= results["target_rate_control"]
    assert results["target_rate_treated_lower"] <= results["target_rate_treated"]
    assert results["ate_lower"] <= results["ate"]

    # check that upper bound is not lower than point estimate
    assert results["target_rate_control"] <= results["target_rate_control_upper"]
    assert results["target_rate_treated"] <= results["target_rate_treated_upper"]
    assert results["ate"] <= results["ate_upper"]

    # test if the point estimates remain accurate
    assert results["target_rate_control"] == pytest.approx(actuals.loc[0])
    assert results["target_rate_treated"] == pytest.approx(actuals.loc[1])
    assert results["ate"] == pytest.approx(actuals.loc[1] - actuals.loc[0])


def test_compute_ate_binary(spark: SparkSession, df_spark_binary: pyspark.sql.DataFrame) -> None:
    actuals = (
        df_spark_binary.groupby("treatment")
        .agg(F.mean("outcome").alias("target_rate"))
        .toPandas()
        .set_index("treatment")
    )

    # Test w/o bootstrapping #
    results = estimate_ate(df_spark_binary, bootstrap=False)
    outcome_rate_control = results["target_rate_control"]
    outcome_rate_treated = results["target_rate_treated"]
    ate = results["ate"]

    assert outcome_rate_control == pytest.approx(actuals.loc[0])
    assert outcome_rate_treated == pytest.approx(actuals.loc[1])
    assert ate == pytest.approx(actuals.loc[1] - actuals.loc[0])

    # Test with bootstrapping #
    results = estimate_ate(df_spark_binary, bootstrap=True, n_bootstraps=100, ci_quantiles=[0.1, 0.9])

    assert len(results) == 9

    # check that lower bound is not higher than point estimate
    assert results["target_rate_control_lower"] <= results["target_rate_control"]
    assert results["target_rate_treated_lower"] <= results["target_rate_treated"]
    assert results["ate_lower"] <= results["ate"]

    # check that upper bound is not lower than point estimate
    assert results["target_rate_control"] <= results["target_rate_control_upper"]
    assert results["target_rate_treated"] <= results["target_rate_treated_upper"]
    assert results["ate"] <= results["ate_upper"]

    # test if the point estimates remain accurate
    assert results["target_rate_control"] == pytest.approx(actuals.loc[0])
    assert results["target_rate_treated"] == pytest.approx(actuals.loc[1])
    assert results["ate"] == pytest.approx(actuals.loc[1] - actuals.loc[0])


def test_compute_target_rate_per_bucket(spark: SparkSession, df_spark_cont: pyspark.sql.DataFrame) -> None:
    df_spark_with_bucket = df_spark_cont.withColumn("bucket", F.when(F.col("feature_1") > 0, "b1").otherwise("b2"))

    actuals = (
        df_spark_with_bucket.groupby("bucket")
        .agg(F.mean("outcome").alias("target_rate"), F.count("*").alias("count"))
        .toPandas()
    )
    actuals["fraction"] = actuals["count"] / actuals["count"].sum()

    # Test w/o bootstrapping #
    df_target_rate = estimate_target_rate_per_bucket(df_spark_with_bucket, bootstrap=False)
    assert df_target_rate.shape == (2, 4)
    merged = df_target_rate.merge(actuals, on="bucket", suffixes=("", "_actual"))
    assert merged.shape == (2, 7)
    assert all(merged["count"] == merged["count_actual"])
    assert all(np.abs(merged["target_rate"] - merged["target_rate_actual"]) < 0.01)
    assert all(np.abs(merged["fraction"] - merged["fraction_actual"]) < 0.01)

    # Test with bootstrapping #
    df_target_rate = estimate_target_rate_per_bucket(
        df_spark_with_bucket, bootstrap=True, n_bootstraps=100, ci_quantiles=[0.1, 0.9]
    )
    assert df_target_rate.shape == (2, 7)
    merged = df_target_rate.merge(actuals, on="bucket", suffixes=("", "_actual"))
    assert merged.shape == (2, 10)
    assert all(merged["count"] == merged["count_actual"])
    assert all(np.abs(merged["target_rate"] - merged["target_rate_actual"]) < 0.01)
    assert all(np.abs(merged["fraction"] - merged["fraction_actual"]) < 0.01)

    # check that lower bound is not higher than point estimate
    assert all(df_target_rate["target_rate_upper"] >= df_target_rate["target_rate"])
    assert all(df_target_rate["target_rate_lower"] <= df_target_rate["target_rate"])


def test_compute_target_rate_per_quantile(spark: SparkSession, df_spark_cont: pyspark.sql.DataFrame) -> None:
    n_users = df_spark_cont.count()
    overall_target_rate = df_spark_cont.select(F.mean("outcome")).toPandas().squeeze()
    n_buckets = 5

    # Test w/o bootstrapping #
    df_target_rate = estimate_target_rate_per_quantile(
        df_spark_cont, n_buckets=n_buckets, score_colname="feature_1", bootstrap=False
    )
    assert df_target_rate.shape == (n_buckets, 4)
    assert df_target_rate["count"].sum() == n_users
    assert df_target_rate["fraction"].sum() == pytest.approx(1)
    assert all(np.abs(df_target_rate["count"] - n_users / n_buckets) < 2)
    assert (df_target_rate["fraction"] * df_target_rate["target_rate"]).sum() == pytest.approx(overall_target_rate)

    # Test with bootstrapping #
    df_target_rate = estimate_target_rate_per_quantile(
        df_spark_cont,
        n_buckets=n_buckets,
        score_colname="feature_1",
        bootstrap=True,
        n_bootstraps=100,
        ci_quantiles=[0.1, 0.9],
    )
    assert df_target_rate.shape == (n_buckets, 7)
    assert df_target_rate["count"].sum() == n_users
    assert df_target_rate["fraction"].sum() == pytest.approx(1)
    assert all(np.abs(df_target_rate["count"] - n_users / n_buckets) < 2)
    assert (df_target_rate["fraction"] * df_target_rate["target_rate"]).sum() == pytest.approx(overall_target_rate)

    # check that lower bound is not higher than point estimate
    assert all(df_target_rate["target_rate_upper"] >= df_target_rate["target_rate"])
    assert all(df_target_rate["target_rate_lower"] <= df_target_rate["target_rate"])


def test_compute_cate_per_bucket(spark: SparkSession, df_spark_binary: pyspark.sql.DataFrame) -> None:
    df_spark_with_bucket = df_spark_binary.withColumn("bucket", F.when(F.col("feature_1") > 0, "b1").otherwise("b2"))

    actuals = (
        df_spark_with_bucket.groupby("bucket", "treatment")
        .agg(F.mean("outcome").alias("target_rate"), F.count("*").alias("count"))
        .toPandas()
    )
    users = actuals.groupby("bucket")["count"].sum().reset_index()
    cates = (
        actuals.set_index("treatment").groupby("bucket")["target_rate"].agg(lambda x: x.loc[1] - x.loc[0]).reset_index()
    )
    cates = cates.rename(columns={"target_rate": "cate"})
    actuals = users.merge(cates, on="bucket")
    actuals["fraction"] = actuals["count"] / actuals["count"].sum()

    # Test w/o bootstrapping #
    df_cate = estimate_cate_per_bucket(df_spark_with_bucket, bootstrap=False)
    assert df_cate.shape == (2, 4)
    merged = df_cate.merge(actuals, on="bucket", suffixes=("", "_actual"))
    assert merged.shape == (2, 7)
    assert all(merged["count"] == merged["count_actual"])
    assert all(np.abs(merged["cate"] - merged["cate_actual"]) < 0.01)
    assert all(np.abs(merged["fraction"] - merged["fraction_actual"]) < 0.01)

    # Test with bootstrapping #
    df_cate = estimate_cate_per_bucket(df_spark_with_bucket, bootstrap=True, n_bootstraps=100, ci_quantiles=[0.1, 0.9])
    assert df_cate.shape == (2, 7)
    merged = df_cate.merge(actuals, on="bucket", suffixes=("", "_actual"))
    assert merged.shape == (2, 10)
    assert all(merged["count"] == merged["count_actual"])
    assert all(np.abs(merged["cate"] - merged["cate_actual"]) < 0.01)
    assert all(np.abs(merged["fraction"] - merged["fraction_actual"]) < 0.01)

    # check that lower bound is not higher than point estimate
    assert all(df_cate["cate_upper"] >= df_cate["cate"])
    assert all(df_cate["cate_lower"] <= df_cate["cate"])


def test_compute_cate_per_quantile(spark: SparkSession, df_spark_binary: pyspark.sql.DataFrame) -> None:
    n_users = df_spark_binary.count()
    n_buckets = 5

    # Test w/o bootstrapping #
    df_cate = estimate_cate_per_quantile(
        df_spark_binary, n_buckets=n_buckets, score_colname="feature_1", bootstrap=False
    )
    assert df_cate.shape == (n_buckets, 4)
    assert df_cate["count"].sum() == n_users
    assert df_cate["fraction"].sum() == pytest.approx(1)
    assert all(np.abs(df_cate["count"] - n_users / n_buckets) < 2)
    assert all(df_cate["cate"] >= -1)
    assert all(df_cate["cate"] <= 1)

    # Test with bootstrapping #
    df_cate = estimate_cate_per_quantile(
        df_spark_binary,
        n_buckets=n_buckets,
        score_colname="feature_1",
        bootstrap=True,
        n_bootstraps=100,
        ci_quantiles=[0.1, 0.9],
    )
    assert df_cate.shape == (n_buckets, 7)
    assert df_cate["count"].sum() == n_users
    assert df_cate["fraction"].sum() == pytest.approx(1)
    assert all(np.abs(df_cate["count"] - n_users / n_buckets) < 2)
    assert all(df_cate["cate"] >= -1)
    assert all(df_cate["cate"] <= 1)

    # check that lower bound is not higher than point estimate
    assert all(df_cate["cate_upper"] >= df_cate["cate"])
    assert all(df_cate["cate_lower"] <= df_cate["cate"])


def test_compute_cate_lift(spark: SparkSession, df_spark_binary: pyspark.sql.DataFrame) -> None:
    n_users = df_spark_binary.count()
    overall_ate = estimate_ate(df_spark_binary, bootstrap=False)["ate"]
    n_buckets = 5

    # Test w/o bootstrapping #
    df_lift = estimate_cate_lift(df_spark_binary, n_buckets=n_buckets, score_colname="feature_1", bootstrap=False)
    assert df_lift.shape == (n_buckets, 4)
    assert df_lift["count"].max() == n_users
    assert df_lift["fraction"].max() == pytest.approx(1)
    assert all(np.abs(df_lift["count"] - n_users * df_lift["fraction"]) < 2)
    assert all(df_lift["cum_cate"] >= -1)
    assert all(df_lift["cum_cate"] <= 1)
    assert df_lift.query(f"count == {n_users}")["cum_cate"].squeeze() == pytest.approx(overall_ate)

    # Test with bootstrapping #
    df_lift = estimate_cate_lift(
        df_spark_binary,
        n_buckets=n_buckets,
        score_colname="feature_1",
        bootstrap=True,
        n_bootstraps=100,
        ci_quantiles=[0.1, 0.9],
    )
    assert df_lift.shape == (n_buckets, 7)
    assert df_lift["count"].max() == n_users
    assert df_lift["fraction"].max() == pytest.approx(1)
    assert all(np.abs(df_lift["count"] - n_users * df_lift["fraction"]) < 2)
    assert all(df_lift["cum_cate"] >= -1)
    assert all(df_lift["cum_cate"] <= 1)
    assert df_lift.query(f"count == {n_users}")["cum_cate"].squeeze() == pytest.approx(overall_ate)

    # check that lower bound is not higher than point estimate
    assert all(df_lift["cum_cate_upper"] >= df_lift["cum_cate"])
    assert all(df_lift["cum_cate_lower"] <= df_lift["cum_cate"])


def test_compute_qini(spark: SparkSession, df_spark_binary: pyspark.sql.DataFrame) -> None:
    n_users = df_spark_binary.count()
    overall_ate = estimate_ate(df_spark_binary, bootstrap=False)["ate"]
    n_buckets = 5

    # Test w/o bootstrapping #
    df_qini = estimate_qini(df_spark_binary, n_buckets=n_buckets, score_colname="feature_1", bootstrap=False)
    assert df_qini.shape == (n_buckets + 1, 5)
    assert df_qini["count"].max() == n_users
    assert df_qini["fraction"].max() == pytest.approx(1)
    assert all(np.abs(df_qini["count"] - n_users * df_qini["fraction"]) < 2)
    assert all(df_qini["ate"] >= -1)
    assert all(df_qini["ate"] <= 1)
    assert df_qini.query(f"count == {n_users}")["ate"].squeeze() == pytest.approx(overall_ate)

    # Test with bootstrapping #
    df_qini = estimate_qini(
        df_spark_binary,
        n_buckets=n_buckets,
        score_colname="feature_1",
        bootstrap=True,
        n_bootstraps=100,
        ci_quantiles=[0.1, 0.9],
    )
    assert df_qini.shape == (n_buckets + 1, 8)
    assert df_qini["count"].max() == n_users
    assert df_qini["fraction"].max() == pytest.approx(1)
    assert all(np.abs(df_qini["count"] - n_users * df_qini["fraction"]) < 2)
    assert all(df_qini["ate"] >= -1)
    assert all(df_qini["ate"] <= 1)
    assert df_qini.query(f"count == {n_users}")["ate"].squeeze() == pytest.approx(overall_ate)

    # check that lower bound is not higher than point estimate
    assert all(df_qini["ate_upper"] >= df_qini["ate"])
    assert all(df_qini["ate_lower"] <= df_qini["ate"])


def test_compute_qini_coefficient() -> None:
    df_qini = pd.DataFrame(
        {
            "bucket": [None, 2.0, 1.0, 0.0],
            "ate": [0.0, 0.1875, 0.5, 0.5],
            "count": [0, 3, 6, 8],
            "fraction": [0.0, 0.375, 0.75, 1.0],
        }
    )

    qini_score = compute_qini_coefficient(df_qini)
    assert qini_score == pytest.approx(0.0390625)


def test_compute_roi(spark: SparkSession, df_spark_binary_with_cost: pyspark.sql.DataFrame) -> None:
    actual_roi = (
        df_spark_binary_with_cost.select(F.sum("benefit").alias("benefit"), F.sum("cost").alias("cost"))
        .withColumn("roi", F.col("benefit") / F.col("cost"))
        .select("roi")
        .toPandas()
        .squeeze()
    )

    # Test w/o bootstrapping #
    results = estimate_roi(df_spark_binary_with_cost, benefit_colname="benefit", cost_colname="cost", bootstrap=False)
    roi = results["roi"]

    assert roi == pytest.approx(actual_roi)

    # Test with bootstrapping #
    results = estimate_roi(
        df_spark_binary_with_cost,
        benefit_colname="benefit",
        cost_colname="cost",
        bootstrap=True,
        n_bootstraps=100,
        ci_quantiles=[0.1, 0.9],
    )
    assert len(results) == 3

    assert results["roi"] == pytest.approx(actual_roi)
    assert results["roi_lower"] <= results["roi"]
    assert results["roi"] <= results["roi_upper"]


def test_compute_iroi(spark: SparkSession, df_spark_binary_with_cost: pyspark.sql.DataFrame) -> None:
    actuals = (
        df_spark_binary_with_cost.groupby("treatment")
        .agg(F.sum("benefit").alias("benefit"), F.sum("cost").alias("cost"), F.count("*").alias("count"))
        .toPandas()
        .set_index("treatment")
    )
    scale_factor = actuals.loc[1]["count"] / actuals.loc[0]["count"]
    actuals.loc[0] *= scale_factor
    actuals_incremental = actuals.loc[1] - actuals.loc[0]

    # Test w/o bootstrapping #
    results = estimate_iroi(df_spark_binary_with_cost, benefit_colname="benefit", cost_colname="cost", bootstrap=False)
    assert len(results) == 3
    iroi = results["iroi"]
    incremental_benefit = results["incremental_benefit"]
    incremental_cost = results["incremental_cost"]

    assert incremental_benefit == pytest.approx(actuals_incremental["benefit"])
    assert incremental_cost == pytest.approx(actuals_incremental["cost"])
    assert iroi == pytest.approx(actuals_incremental["benefit"] / actuals_incremental["cost"])

    # Test with bootstrapping #
    results = estimate_iroi(
        df_spark_binary_with_cost,
        benefit_colname="benefit",
        cost_colname="cost",
        bootstrap=True,
        n_bootstraps=100,
        ci_quantiles=[0.1, 0.9],
    )
    assert len(results) == 9

    # check that lower bound is not higher than point estimate
    assert results["incremental_benefit_lower"] <= results["incremental_benefit"]
    assert results["incremental_cost_lower"] <= results["incremental_cost"]
    assert results["iroi_lower"] <= results["iroi"]

    # check that upper bound is not lower than point estimate
    assert results["incremental_benefit_upper"] >= results["incremental_benefit"]
    assert results["incremental_cost_upper"] >= results["incremental_cost"]
    assert results["iroi_upper"] >= results["iroi"]

    # test if the point estimates remain accurate
    assert results["incremental_benefit"] == pytest.approx(actuals_incremental["benefit"])
    assert results["incremental_cost"] == pytest.approx(actuals_incremental["cost"])
    assert results["iroi"] == pytest.approx(actuals_incremental["benefit"] / actuals_incremental["cost"])


def test_compute_iroi_per_bucket(spark: SparkSession, df_spark_binary_with_cost: pyspark.sql.DataFrame) -> None:
    df_spark_with_bucket = df_spark_binary_with_cost.withColumn(
        "bucket", F.when(F.col("feature_1") > 0, "b1").otherwise("b2")
    )

    actuals = (
        df_spark_with_bucket.groupby("bucket", "treatment")
        .agg(F.sum("benefit").alias("benefit"), F.sum("cost").alias("cost"), F.count("*").alias("count"))
        .toPandas()
    )
    users = actuals.groupby("bucket")["count"].sum().reset_index()
    actuals = actuals.set_index(["treatment", "bucket"])
    scale_factor = actuals.loc[1]["count"] / actuals.loc[0]["count"]
    actuals.loc[0, "b1"] = actuals.loc[0, "b1"] * scale_factor["b1"]
    actuals.loc[0, "b2"] = actuals.loc[0, "b2"] * scale_factor["b2"]
    actuals_incremental = (
        actuals.groupby("bucket")[["benefit", "cost"]].agg(lambda x: x.loc[1] - x.loc[0]).reset_index()
    )
    actuals_incremental["iroi"] = actuals_incremental["benefit"] / actuals_incremental["cost"]
    actuals = users.merge(actuals_incremental.drop(["cost", "benefit"], axis=1), on="bucket")
    actuals["fraction"] = actuals["count"] / actuals["count"].sum()

    # Test w/o bootstrapping #
    df_iroi = estimate_iroi_per_bucket(
        df_spark_with_bucket, benefit_colname="benefit", cost_colname="cost", bootstrap=False
    )
    assert df_iroi.shape == (2, 4)
    merged = df_iroi.merge(actuals, on="bucket", suffixes=("", "_actual"))
    assert merged.shape == (2, 7)
    assert all(merged["count"] == merged["count_actual"])
    assert all(np.abs(merged["iroi"] - merged["iroi_actual"]) < 0.01)
    assert all(np.abs(merged["fraction"] - merged["fraction_actual"]) < 0.01)

    # Test with bootstrapping #
    df_iroi = estimate_iroi_per_bucket(
        df_spark_with_bucket,
        benefit_colname="benefit",
        cost_colname="cost",
        bootstrap=True,
        n_bootstraps=100,
        ci_quantiles=[0.1, 0.9],
    )
    assert df_iroi.shape == (2, 7)
    merged = df_iroi.merge(actuals, on="bucket", suffixes=("", "_actual"))
    assert merged.shape == (2, 10)
    assert all(merged["count"] == merged["count_actual"])
    assert all(np.abs(merged["iroi"] - merged["iroi_actual"]) < 0.01)
    assert all(np.abs(merged["fraction"] - merged["fraction_actual"]) < 0.01)

    # check that lower bound is not higher than point estimate
    assert all(df_iroi["iroi_upper"] >= df_iroi["iroi"])
    assert all(df_iroi["iroi_lower"] <= df_iroi["iroi"])


def test_compute_iroi_per_quantile(spark: SparkSession, df_spark_binary_with_cost: pyspark.sql.DataFrame) -> None:
    n_users = df_spark_binary_with_cost.count()
    n_buckets = 5

    # Test w/o bootstrapping #
    df_iroi = estimate_iroi_per_quantile(
        df_spark_binary_with_cost,
        n_buckets=n_buckets,
        score_colname="feature_1",
        benefit_colname="benefit",
        cost_colname="cost",
        bootstrap=False,
    )
    assert df_iroi.shape == (n_buckets, 4)
    assert df_iroi["count"].sum() == n_users
    assert df_iroi["fraction"].sum() == pytest.approx(1)
    assert all(np.abs(df_iroi["count"] - n_users / n_buckets) < 2)

    # Test with bootstrapping #
    df_iroi = estimate_iroi_per_quantile(
        df_spark_binary_with_cost,
        n_buckets=n_buckets,
        score_colname="feature_1",
        benefit_colname="benefit",
        cost_colname="cost",
        bootstrap=True,
        n_bootstraps=100,
        ci_quantiles=[0.1, 0.9],
    )
    assert df_iroi.shape == (n_buckets, 7)
    assert df_iroi["count"].sum() == n_users
    assert df_iroi["fraction"].sum() == pytest.approx(1)
    assert all(np.abs(df_iroi["count"] - n_users / n_buckets) < 2)

    # check that lower bound is not higher than point estimate
    assert all(df_iroi["iroi_upper"] >= df_iroi["iroi"])
    assert all(df_iroi["iroi_lower"] <= df_iroi["iroi"])


def test_compute_cum_iroi(spark: SparkSession, df_spark_binary_with_cost: pyspark.sql.DataFrame) -> None:
    n_users = df_spark_binary_with_cost.count()
    iroi_actual = estimate_iroi(
        df_spark_binary_with_cost, benefit_colname="benefit", cost_colname="cost", bootstrap=False
    )["iroi"]
    n_buckets = 5

    # Test w/o bootstrapping #
    df_iroi = estimate_cum_iroi(
        df_spark_binary_with_cost,
        n_buckets=n_buckets,
        score_colname="feature_1",
        benefit_colname="benefit",
        cost_colname="cost",
        bootstrap=False,
    )
    assert df_iroi.shape == (n_buckets, 5)
    assert df_iroi["count"].max() == n_users
    assert df_iroi["fraction"].max() == pytest.approx(1)
    assert all(np.abs(df_iroi["count"] - n_users * df_iroi["fraction"]) < 2)
    assert df_iroi.query(f"count == {n_users}")["iroi"].squeeze() == pytest.approx(iroi_actual)

    # Test with bootstrapping #
    df_iroi = estimate_cum_iroi(
        df_spark_binary_with_cost,
        n_buckets=n_buckets,
        score_colname="feature_1",
        benefit_colname="benefit",
        cost_colname="cost",
        bootstrap=True,
        n_bootstraps=100,
        ci_quantiles=[0.1, 0.9],
    )
    assert df_iroi.shape == (n_buckets, 8)
    assert df_iroi["count"].max() == n_users
    assert df_iroi["fraction"].max() == pytest.approx(1)
    assert all(np.abs(df_iroi["count"] - n_users * df_iroi["fraction"]) < 2)
    assert df_iroi.query(f"count == {n_users}")["iroi"].squeeze() == pytest.approx(iroi_actual)

    # check that lower bound is not higher than point estimate
    assert all(df_iroi["iroi_upper"] >= df_iroi["iroi"])
    assert all(df_iroi["iroi_lower"] <= df_iroi["iroi"])
