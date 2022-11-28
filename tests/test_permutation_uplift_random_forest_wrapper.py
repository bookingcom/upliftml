import h2o  # type: ignore
import pyspark  # type: ignore
from pyspark.sql import SparkSession

from upliftml.feature_selection.stationary_methods import (
    PermutationUpliftRandomForestWrapper,
)


def test_calculate_feature_importance(spark: SparkSession, df_spark_binary: pyspark.sql.DataFrame) -> None:
    df_spark_binary = df_spark_binary.drop("propensity", "expected_outcome", "actual_cate")
    features = [col for col in df_spark_binary.columns if "feature" in col]

    categorical_features = ["treatment", "outcome"]

    # Hyperparameter for Distributed Uplift Random Forest
    durf_dict = {
        "ntrees": 5,
        "max_depth": 5,
        "treatment_column": "treatment",
        "uplift_metric": "kl",  # Can either be 'kl', 'euclidean', or 'chi_squared'
        "nbins_top_level": 512,
        "nbins": 10,
        "min_rows": 1,
        "mtries": -1,
        "sample_rate": 0.5,
        "sample_rate_per_class": None,
        "col_sample_rate_per_tree": 0.5,
        "auuc_type": "qini",
        "auuc_nbins": 100,
        "ignore_const_cols": False,
    }

    permutation_urf = PermutationUpliftRandomForestWrapper(
        durf_dict=durf_dict, factor_list=categorical_features, n_repeats=3
    )

    df_h2o = h2o.H2OFrame(df_spark_binary)
    fi_non_dynamicity_p = permutation_urf.calculate_feature_importance(df_h2o, features)

    # Check if the feature importance dataframe contains as much rows as we have features
    assert fi_non_dynamicity_p.shape[0] == len(features)
    assert fi_non_dynamicity_p.shape[1] == 2
