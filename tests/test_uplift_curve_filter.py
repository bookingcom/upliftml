import h2o  # type: ignore
import pyspark  # type: ignore
from pyspark.sql import SparkSession

from upliftml.feature_selection.stationary_methods import UpliftCurveFilter


def test_calculate_feature_importance(spark: SparkSession, df_spark_binary: pyspark.sql.DataFrame) -> None:
    df_spark_binary = df_spark_binary.drop("propensity", "expected_outcome", "actual_cate")
    features = [col for col in df_spark_binary.columns if "feature" in col]
    n_bins = 10

    uplift_curve = UpliftCurveFilter(n_bins=n_bins)
    fi_uc, fi_variables_uc = uplift_curve.calculate_feature_importance(df_spark_binary, features)

    # Check if the feature importance dataframe contains as much rows as we have features
    assert fi_uc.shape[0] == len(features)
    assert fi_uc.shape[1] == 2

    # Check if the feature importance variables dictionary contains as much keys as we have features
    assert len(fi_variables_uc) == len(features)
    # For each feature, check if the length of values equals n_bins
    for k, v in fi_variables_uc.items():
        assert len(v) == n_bins
