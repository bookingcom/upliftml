import h2o  # type: ignore
from h2o.estimators.glm import H2OGeneralizedLinearEstimator  # type: ignore
from pyspark.sql import SparkSession
from sklearn.metrics import roc_auc_score  # type: ignore

from upliftml.models.h2o import RetrospectiveEstimator


def test_train_model_binary_outcome(spark: SparkSession, df_h2o_binary: h2o.H2OFrame) -> None:
    orig_data_types = df_h2o_binary.types
    orig_shape = df_h2o_binary.shape

    predictor_colnames = [col for col in df_h2o_binary.columns if col.startswith("feature")]

    model = RetrospectiveEstimator(
        base_model_class=H2OGeneralizedLinearEstimator,
        base_model_params={},
        predictor_colnames=predictor_colnames,
        target_colname="outcome",
        treatment_colname="treatment",
        positive_outcome_value=1,
    )

    model.fit(df_h2o_binary)

    df_h2o_train_eval = df_h2o_binary.cbind(model.predict(df_h2o_binary).set_names(["predicted_score"]))
    df_pd_train_eval = df_h2o_train_eval.as_data_frame()

    # check that after fitting the model and predicting, the df_h2o still has the same shape as the original df_pd
    assert df_h2o_binary.shape == orig_shape

    # check that the evaluation dataset has the expected shape
    assert df_h2o_train_eval.shape[0] == orig_shape[0]
    assert df_h2o_train_eval.shape[1] == orig_shape[1] + 1

    # check that the column types in df_h2o still match the original ones
    assert df_h2o_binary.types == orig_data_types

    # check that the predictions are better than random
    df_pd_train_eval_bookers = df_pd_train_eval[df_pd_train_eval["outcome"] == 1]
    assert roc_auc_score(df_pd_train_eval_bookers["treatment"], df_pd_train_eval_bookers["predicted_score"]) > 0.5

    # check that the predicted treatment effect values are between [-1, 1]
    assert df_pd_train_eval["predicted_score"].min() >= 0
    assert df_pd_train_eval["predicted_score"].max() <= 1


def test_train_model_same_train_val_set(spark: SparkSession, df_h2o_binary: h2o.H2OFrame) -> None:
    orig_data_types = df_h2o_binary.types
    orig_shape = df_h2o_binary.shape

    predictor_colnames = [col for col in df_h2o_binary.columns if col.startswith("feature")]

    model = RetrospectiveEstimator(
        base_model_class=H2OGeneralizedLinearEstimator,
        base_model_params={},
        predictor_colnames=predictor_colnames,
        target_colname="outcome",
        treatment_colname="treatment",
        positive_outcome_value=1,
    )

    model.fit(df_h2o_binary, df_h2o_binary)

    df_h2o_train_eval = df_h2o_binary.cbind(model.predict(df_h2o_binary).set_names(["predicted_score"]))
    df_pd_train_eval = df_h2o_train_eval.as_data_frame()

    # check that after fitting the model and predicting, the df_h2o still has the same shape as the original df_pd
    assert df_h2o_binary.shape == orig_shape

    # check that the evaluation dataset has the expected shape
    assert df_h2o_train_eval.shape[0] == orig_shape[0]
    assert df_h2o_train_eval.shape[1] == orig_shape[1] + 1

    # check that the column types in df_h2o still match the original ones
    assert df_h2o_binary.types == orig_data_types

    # check that the predictions are better than random
    df_pd_train_eval_bookers = df_pd_train_eval[df_pd_train_eval["outcome"] == 1]
    assert roc_auc_score(df_pd_train_eval_bookers["treatment"], df_pd_train_eval_bookers["predicted_score"]) > 0.5

    # check that the predicted treatment effect values are between [-1, 1]
    assert df_pd_train_eval["predicted_score"].min() >= 0
    assert df_pd_train_eval["predicted_score"].max() <= 1


def test_train_model_val_set(spark: SparkSession, df_h2o_binary: h2o.H2OFrame, df_h2o_binary_val: h2o.H2OFrame) -> None:
    orig_data_types = df_h2o_binary.types
    orig_shape = df_h2o_binary.shape

    orig_data_types_val = df_h2o_binary_val.types
    orig_shape_val = df_h2o_binary_val.shape

    predictor_colnames = [col for col in df_h2o_binary.columns if col.startswith("feature")]

    model = RetrospectiveEstimator(
        base_model_class=H2OGeneralizedLinearEstimator,
        base_model_params={},
        predictor_colnames=predictor_colnames,
        target_colname="outcome",
        treatment_colname="treatment",
        positive_outcome_value=1,
    )

    model.fit(df_h2o_binary, df_h2o_binary_val)

    df_h2o_train_eval = df_h2o_binary.cbind(model.predict(df_h2o_binary).set_names(["predicted_score"]))
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
    df_pd_train_eval_bookers = df_pd_train_eval[df_pd_train_eval["outcome"] == 1]
    assert roc_auc_score(df_pd_train_eval_bookers["treatment"], df_pd_train_eval_bookers["predicted_score"]) > 0.5

    # check that the predicted treatment effect values are between [-1, 1]
    assert df_pd_train_eval["predicted_score"].min() >= 0
    assert df_pd_train_eval["predicted_score"].max() <= 1
