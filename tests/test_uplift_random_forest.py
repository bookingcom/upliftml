import h2o  # type: ignore
from pyspark.sql import SparkSession
from sklearn.metrics import r2_score  # type: ignore

from upliftml.models.h2o import UpliftRandomForestEstimator


def test_train_model_binary_outcome(spark: SparkSession, df_h2o_binary: h2o.H2OFrame) -> None:
    orig_data_types = df_h2o_binary.types
    orig_shape = df_h2o_binary.shape

    predictor_colnames = [col for col in df_h2o_binary.columns if col.startswith("feature")]

    base_model_params = {
        "treatment_column": "treatment",
        "uplift_metric": "kl",
        "auuc_type": "qini",
        "auuc_nbins": 10,
        "ntrees": 10,
        "max_depth": 10,
    }

    model = UpliftRandomForestEstimator(
        base_model_params=base_model_params,
        predictor_colnames=predictor_colnames,
        treatment_colname="treatment",
        target_colname="outcome",
    )

    model.fit(df_h2o_binary)

    df_h2o_train_eval = df_h2o_binary.cbind(model.predict(df_h2o_binary).set_names(["predicted_cate"]))
    df_pd_train_eval = df_h2o_train_eval.as_data_frame()

    # check that after fitting the model and predicting, the df_h2o still has the same shape as the original df_pd
    assert df_h2o_binary.shape == orig_shape

    # check that the evaluation dataset has the expected shape
    assert df_h2o_train_eval.shape[0] == orig_shape[0]
    assert df_h2o_train_eval.shape[1] == orig_shape[1] + 1

    # check that the column types in df_h2o still match the original ones
    assert df_h2o_binary.types == orig_data_types

    # check that the predictions are better than random
    assert r2_score(df_pd_train_eval["actual_cate"], df_pd_train_eval["predicted_cate"]) > 0


def test_train_model_same_train_val_set(spark: SparkSession, df_h2o_binary: h2o.H2OFrame) -> None:
    orig_data_types = df_h2o_binary.types
    orig_shape = df_h2o_binary.shape

    predictor_colnames = [col for col in df_h2o_binary.columns if col.startswith("feature")]

    base_model_params = {
        "treatment_column": "treatment",
        "uplift_metric": "kl",
        "auuc_type": "qini",
        "auuc_nbins": 10,
        "ntrees": 10,
        "max_depth": 10,
    }

    model = UpliftRandomForestEstimator(
        base_model_params=base_model_params,
        predictor_colnames=predictor_colnames,
        treatment_colname="treatment",
        target_colname="outcome",
    )

    model.fit(df_h2o_binary, df_h2o_binary)

    df_h2o_train_eval = df_h2o_binary.cbind(model.predict(df_h2o_binary).set_names(["predicted_cate"]))
    df_pd_train_eval = df_h2o_train_eval.as_data_frame()

    # check that after fitting the model and predicting, the df_h2o still has the same shape as the original df_pd
    assert df_h2o_binary.shape == orig_shape

    # check that the evaluation dataset has the expected shape
    assert df_h2o_train_eval.shape[0] == orig_shape[0]
    assert df_h2o_train_eval.shape[1] == orig_shape[1] + 1

    # check that the column types in df_h2o still match the original ones
    assert df_h2o_binary.types == orig_data_types

    # check that the predictions are better than random
    assert r2_score(df_pd_train_eval["actual_cate"], df_pd_train_eval["predicted_cate"]) > 0


def test_train_model_val_set(spark: SparkSession, df_h2o_binary: h2o.H2OFrame, df_h2o_binary_val: h2o.H2OFrame) -> None:
    orig_data_types = df_h2o_binary.types
    orig_shape = df_h2o_binary.shape

    orig_data_types_val = df_h2o_binary_val.types
    orig_shape_val = df_h2o_binary_val.shape

    predictor_colnames = [col for col in df_h2o_binary.columns if col.startswith("feature")]

    base_model_params = {
        "treatment_column": "treatment",
        "uplift_metric": "kl",
        "auuc_type": "qini",
        "auuc_nbins": 10,
        "ntrees": 10,
        "max_depth": 10,
    }

    model = UpliftRandomForestEstimator(
        base_model_params=base_model_params,
        predictor_colnames=predictor_colnames,
        treatment_colname="treatment",
        target_colname="outcome",
    )

    model.fit(df_h2o_binary, df_h2o_binary_val)

    df_h2o_train_eval = df_h2o_binary.cbind(model.predict(df_h2o_binary).set_names(["predicted_cate"]))
    df_pd_train_eval = df_h2o_train_eval.as_data_frame()

    # check that after fitting the model and predicting, the df_h2o still has the same shape as the original df_pd
    assert df_h2o_binary.shape == orig_shape
    assert df_h2o_binary_val.shape == orig_shape_val

    # check that the evaluation dataset has the expected shape
    assert df_h2o_train_eval.shape[0] == orig_shape[0]
    assert df_h2o_train_eval.shape[1] == orig_shape[1] + 1

    # check that the column types in df_h2o still match the original ones
    assert df_h2o_binary.types == orig_data_types
    assert df_h2o_binary_val.types == orig_data_types_val

    # check that the predictions are better than random
    assert r2_score(df_pd_train_eval["actual_cate"], df_pd_train_eval["predicted_cate"]) > 0
