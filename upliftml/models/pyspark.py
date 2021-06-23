from typing import Any, Dict, List, Optional, Union

import pyspark
import pyspark.sql.functions as F
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType


class SLearnerEstimator:
    """Estimates treatment effect by training a single model for E[Y | T, X], applying the model with T=1 and T=0 and using the difference in these estimates as the estimated treatment effect.

    The name S-learner originates from Künzel et al. (2019) (https://arxiv.org/pdf/1706.03461.pdf).
    """

    def __init__(
        self,
        base_model_class: Any,
        base_model_params: Dict,
        predictors_colname: str = "features",
        treatment_colname: str = "treatment",
        target_colname: str = "outcome",
        treatment_value: int = 1,
        control_value: int = 0,
        output_colname: str = "score",
    ):

        """Initializes the S-learner.

        Args:
            base_model_class (pyspark.ml): the MLlib model class to instantiate the S-learner with
            base_model_params (dict): parameters and their values for the model
            predictors_colname (str, optional): the column names that contain the predictor variables
            treatment_colname (str, optional): the column name that contains the treatment indicators
            target_colname (str, optional): the column name that contains the target
            treatment_value (str or int, optional): the value in column <treatment_colname> that refers to the treatment group
            control_value (str or int, optional): the value in column <treatment_colname> that refers to the control group
            output_colname (str, optional): the column name for the estimator output
        """

        base_model_params["featuresCol"] = "features_train"
        base_model_params["labelCol"] = target_colname

        self.predictors_colname = predictors_colname
        self.model = base_model_class(**base_model_params)
        self.treatment_colname = treatment_colname
        self.target_colname = target_colname
        self.control_value = control_value
        self.treatment_value = treatment_value
        self.output_colname = output_colname

    def fit(self, df_train: pyspark.sql.DataFrame, df_val: Optional[Any] = None) -> None:
        """Trains the S-learner.

        Args:
            df_train (pyspark.sql.DataFrame): a dataframe containing the treatment indicators, the observed outcomes, and predictors
            df_val (pyspark.sql.DataFrame): a dataframe containing the treatment indicators, the observed outcomes, and predictors
        """
        add_treatment_udf = udf(lambda v, e: Vectors.dense(list(v) + [e]), VectorUDT())

        df_train = df_train.withColumn(
            "features_train", add_treatment_udf(self.predictors_colname, self.treatment_colname)
        )
        self.model = self.model.fit(df_train)

    def predict(self, df: pyspark.sql.DataFrame) -> pyspark.sql.DataFrame:
        """Applies the S-learner and returns treatment effect predictions.

        Args:
            df (pyspark.sql.DataFrame): a dataframe containing predictors

        Returns:
            df (pyspark.sql.DataFrame): a dataframe containing predictors containing treatment effect predictions
        """

        df = df.withColumn("original_treatment", F.col(self.treatment_colname))
        df_cols = df.columns
        df = df.withColumn(self.treatment_colname, F.lit(self.control_value))
        add_treatment_udf = udf(lambda v, e: Vectors.dense(list(v) + [e]), VectorUDT())
        df = df.withColumn("features_train", add_treatment_udf(self.predictors_colname, self.treatment_colname))
        df = self.model.transform(df)
        if "probability" not in df.columns:
            split_udf = udf(lambda value: value, FloatType())
            pred_col = "prediction"
        else:
            split_udf = udf(lambda value: value[1].item(), FloatType())
            pred_col = "probability"

        df = df.withColumn("pred_control", split_udf(pred_col)).select([*df_cols, "pred_control"])
        df = df.withColumn(self.treatment_colname, F.lit(self.treatment_value))
        df = df.withColumn("features_train", add_treatment_udf(self.predictors_colname, self.treatment_colname))
        df = (
            self.model.transform(df)
            .withColumn("pred_treatment", split_udf(pred_col))
            .select([*df_cols, "pred_control", "pred_treatment"])
        )

        df = df.withColumn(self.output_colname, F.col("pred_treatment") - F.col("pred_control"))
        df = df.withColumn(self.treatment_colname, F.col("original_treatment"))

        df = df.select([*df_cols, self.output_colname]).drop("original_treatment")

        return df


class TLearnerEstimator:
    """Estimates treatment effect as the difference in estimates from two separate models: E[Y | T=1, X] - E[Y | T=0, X].

    The two-model approach is widely used for treatment effect estimation. The name T-learner originates from Künzel et al. (2019) (https://arxiv.org/pdf/1706.03461.pdf).
    """

    def __init__(
        self,
        base_model_class: Any,
        base_model_params: Dict,
        predictors_colname: str = "features",
        treatment_colname: str = "treatment",
        target_colname: str = "outcome",
        treatment_value: int = 1,
        control_value: int = 0,
        output_colname: str = "score",
    ):

        """Initializes the T-learner.

        Args:
            base_model_class (pyspark.ml): the model class to instantiate the T-learner with
            base_model_params (dict): parameters and their values for the models
            predictors_colname (str, optional): the column names that contain the predictor variables
            treatment_colname (str, optional): the column name that contains the treatment indicators
            target_colname (str, optional): the column name that contains the target
            treatment_value (str or int, optional): the value in column <treatment_colname> that refers to the treatment group
            control_value (str or int, optional): the value in column <treatment_colname> that refers to the control group
            output_colname (str, optional): the column name for the estimator output
        """

        base_model_params["featuresCol"] = predictors_colname
        base_model_params["labelCol"] = target_colname

        self.model0 = base_model_class(**base_model_params)
        self.model1 = base_model_class(**base_model_params)
        self.treatment_colname = treatment_colname
        self.target_colname = target_colname
        self.control_value = control_value
        self.treatment_value = treatment_value
        self.output_colname = output_colname

    def fit(self, df_train: pyspark.sql.DataFrame, df_val: Optional[Any] = None) -> None:
        """Trains the T-learner.

        Args:
            df_train (pyspark.sql.DataFrame): a dataframe containing the treatment indicators, the observed outcomes, and predictors
            df_val (pyspark.sql.DataFrame): a dataframe containing the treatment indicators, the observed outcomes, and predictors
        """
        self.model1 = self.model1.fit(df_train.where(F.col(self.treatment_colname) == self.treatment_value))
        self.model0 = self.model0.fit(df_train.where(F.col(self.treatment_colname) == self.control_value))

    def predict(self, df: pyspark.sql.DataFrame) -> pyspark.sql.DataFrame:
        """Applies the T-learner and returns treatment effect predictions.

        Args:
            df (pyspark.sql.DataFrame): a dataframe containing predictors

        Returns:
            df (pyspark.sql.DataFrame): a dataframe containing treatment effect predictions
        """

        df_cols = df.columns
        df = self.model0.transform(df)
        if "probability" not in df.columns:
            split_udf = udf(lambda value: value, FloatType())
            pred_col = "prediction"
        else:
            split_udf = udf(lambda value: value[1].item(), FloatType())
            pred_col = "probability"

        df = df.withColumn("pred_control", split_udf(pred_col)).select([*df_cols, "pred_control"])
        df = (
            self.model1.transform(df)
            .withColumn("pred_treatment", split_udf(pred_col))
            .select([*df_cols, "pred_control", "pred_treatment"])
        )
        df = df.withColumn(self.output_colname, F.col("pred_treatment") - F.col("pred_control"))

        df = df.select([*df_cols, self.output_colname])

        return df


class XLearnerEstimator:
    """Estimates treatment effect in three stages:
    1. Train a T-learner to get scores Y_hat_1 and Y_hat_0.
    2. Train regression models to predict the residuals: tau1 = E[Y(1) - Y_hat_1) | X] and tau0 = E[Y_hat_0 - Y(0) | X]
    3. Estimate the treatment effect as a weighted average: tau(X) = p(X) * tau0(X) + (1 - p(X)) * tau1(X). Our implementation sets p(X) = 0.5 for all X.

    X-learner was proposed in Künzel et al. (2019) (https://arxiv.org/pdf/1706.03461.pdf).
    """

    def __init__(
        self,
        base_model_class_1: Any,
        base_model_params_1: Dict,
        base_model_class_2: Any,
        base_model_params_2: Dict,
        predictors_colname_2: str = "features",
        predictors_colname_1: str = "features",
        treatment_colname: str = "treatment",
        target_colname: str = "outcome",
        treatment_value: int = 1,
        control_value: int = 0,
        output_colname: str = "score",
    ):

        """Initializes the X-learner.

        Args:
            base_model_class_1 (pyspark.ml): the model class to instantiate the first stage learners
            base_model_params_1 (dict): parameters and their values for the first stage models
            predictors_colname_1 (list of str): the column names that contain the predictor variables for the first stage models
            base_model_class_2 (): the model class to instantiate the second stage learners
            base_model_params_2 (dict): parameters and their values for the second stage models. The models must be regressors.
            predictors_colname_2 (list of str): the column names that contain the predictor variables for the second stage models
            treatment_colname (str, optional): the column name that contains the treatment indicators
            target_colname (str, optional): the column name that contains the target
            treatment_value (str or int, optional): the value in column <treatment_colname> that refers to the treatment group
            control_value (str or int, optional): the value in column <treatment_colname> that refers to the control group
            output_colname (str, optional): the column name for the estimator output
        """

        self.first_learner = TLearnerEstimator(
            base_model_class_1,
            base_model_params_1,
            predictors_colname_1,
            treatment_colname,
            target_colname,
            treatment_value,
            control_value,
        )

        self.second_learner = TLearnerEstimator(
            base_model_class_2,
            base_model_params_2,
            predictors_colname_2,
            treatment_colname,
            "D",
            treatment_value,
            control_value,
        )

        self.treatment_colname = treatment_colname
        self.target_colname = target_colname
        self.control_value = control_value
        self.treatment_value = treatment_value
        self.output_colname = output_colname

    def fit(self, df_train: pyspark.sql.DataFrame, df_val: Optional[Any] = None) -> None:
        """Trains the X-learner.

        Args:
            df_train (pyspark.sql.DataFrame): a dataframe containing the treatment indicators, the observed outcomes, and predictors
            df_val (pyspark.sql.DataFrame): a dataframe containing the treatment indicators, the observed outcomes, and predictors
        """

        self.first_learner.fit(df_train)

        df_train = self._get_second_learner_targets(df_train)

        self.second_learner.fit(df_train)

    def predict(self, df: pyspark.sql.DataFrame) -> pyspark.sql.DataFrame:
        """Applies the X-learner and returns treatment effect predictions.

        Args:
            df (pyspark.sql.DataFrame): a dataframe containing predictors

        Returns:
            df (pyspark.sql.DataFrame): a dataframe containing treatment effect predictions
        """

        df = self._get_second_learner_targets(df)
        df_cols = df.columns
        df = self.second_learner.model0.transform(df)
        if "probability" not in df.columns:
            split_udf = udf(lambda value: value, FloatType())
            pred_col = "prediction"
        else:
            split_udf = udf(lambda value: value[1].item(), FloatType())
            pred_col = "probability"

        df = df.withColumn("pred_control", split_udf(pred_col)).select([*df_cols, "pred_control"])
        df = (
            self.second_learner.model1.transform(df)
            .withColumn("pred_treatment", split_udf(pred_col))
            .select([*df_cols, "pred_control", "pred_treatment"])
        )

        df = df.withColumn(self.output_colname, 0.5 * (F.col("pred_treatment") + F.col("pred_control")))

        df = df.select([*df_cols, self.output_colname]).drop("D")

        return df

    def _get_second_learner_targets(self, df: pyspark.sql.DataFrame) -> pyspark.sql.DataFrame:
        """Applies the first stage models and returns the targets for the second stage model.

        Args:
            df (pyspark.sql.DataFrame): a dataframe containing the treatment indicators, the observed outcomes, and predictors

        Returns:
            df (pyspark.sql.DataFrame): a dataframe containing the second stage targets for the X-learner
        """

        df_cols = df.columns
        df = self.first_learner.model0.transform(df)
        if "probability" not in df.columns:
            split_udf = udf(lambda value: value, FloatType())
            pred_col = "prediction"
        else:
            split_udf = udf(lambda value: value[1].item(), FloatType())
            pred_col = "probability"

        df = df.withColumn("preds0", split_udf(pred_col)).select([*df_cols, "preds0"])
        df = (
            self.first_learner.model1.transform(df)
            .withColumn("preds1", split_udf(pred_col))
            .select([*df_cols, "preds0", "preds1"])
        )

        df = df.withColumn(
            "D",
            F.when(
                F.col(self.treatment_colname) == self.treatment_value, F.col(self.target_colname) - F.col("preds0")
            ).otherwise(F.col("preds1") - F.col(self.target_colname)),
        )

        return df.select([*df_cols, "D"])


class CVTEstimator:
    """Estimates treatment effect by transforming the target variable into a new target variable Z, such that the treatment effect tau(X) = 2 * E[Z | X] - 1.
    This transformation results in a classification problem and is, thus, slightly different from the TransformedOutcomeEstimator, which results in a regression problem.
    Can only be used with 50-50 treatment vs. control RCT data.

    The Class Variable Transformation technique was proposed in Jaskowski and Jaroszewicz (2012) (http://people.cs.pitt.edu/~milos/icml_clinicaldata_2012/Papers/Oral_Jaroszewitz_ICML_Clinical_2012.pdf).
    """

    def __init__(
        self,
        base_model_class: Any,
        base_model_params: Dict,
        predictors_colname: str = "features",
        treatment_colname: str = "treatment",
        target_colname: str = "outcome",
        output_colname: str = "score",
    ):

        """Initializes the CVTEstimator.

        Args:
            base_model_class (pyspark.ml): the model class to instantiate the CVTEstimator with
            base_model_params (dict): parameters and their values for the model
            predictors_colname (str, optional): the column names that contain the predictor variables
            treatment_colname (str, optional): the column name that contains the treatment indicators
            target_colname (str, optional): the column name that contains the target
            output_colname (str, optional): the column name for the estimator output
        """

        base_model_params["featuresCol"] = predictors_colname
        base_model_params["labelCol"] = "cvt_label"

        self.model = base_model_class(**base_model_params)
        self.predictors_colname = predictors_colname
        self.treatment_colname = treatment_colname
        self.target_colname = target_colname
        self.output_colname = output_colname

    def fit(self, df_train: pyspark.sql.DataFrame, df_val: Optional[Any] = None) -> None:
        """Trains the CVT model by transforming the target variable and fitting a classifier on the transformed targets.

        Args:
            df_train (pyspark.sql.DataFrame): a dataframe containing the treatment indicators, the observed outcomes, and predictors
            df_val (pyspark.sql.DataFrame): a dataframe containing the treatment indicators, the observed outcomes, and predictors
        """

        df_train = df_train.withColumn(
            "cvt_label", F.when(F.col(self.treatment_colname) == F.col(self.target_colname), 1).otherwise(0)
        )

        self.model = self.model.fit(df_train)

    def predict(self, df: pyspark.sql.DataFrame) -> pyspark.sql.DataFrame:
        """Applies the CVT model and returns treatment effect predictions.

        Args:
            df (pyspark.sql.DataFrame): a dataframe containing predictors

        Returns:
            df (pyspark.sql.DataFrame): a dataframe containing treatment effect predictions
        """

        df_cols = df.columns
        df = df.withColumn(
            "cvt_label", F.when(F.col(self.treatment_colname) == F.col(self.target_colname), 1).otherwise(0)
        )
        df = self.model.transform(df)
        split_udf = udf(lambda value: value[1].item(), FloatType())
        df = df.withColumn("prob", split_udf("probability"))

        df = df.withColumn(self.output_colname, 2 * F.col("prob") - 1)

        df = df.select([*df_cols, self.output_colname])

        return df


class TransformedOutcomeEstimator:
    """Estimates treatment effect by transforming the outcome, such that the expectation of the transformed outcome corresponds to the treatment effect.
    This transformation results in a regression problem and is, thus, slightly different from the CVTEstimator, which results in a classification problem.

    The Transformed Outcome technique was proposed in Athey and Imbens (2015) (https://pdfs.semanticscholar.org/86ce/004214845a1683d59b64c4363a067d342cac.pdf).
    """

    def __init__(
        self,
        base_model_class: Any,
        base_model_params: Dict,
        predictors_colname: str = "features",
        propensity_model_class: Any = None,
        propensity_model_params: Dict = None,
        treatment_colname: str = "treatment",
        target_colname: str = "outcome",
        treatment_value: int = 1,
        control_value: int = 0,
        output_colname: str = "score",
    ):

        """Initializes the TransformedOutcomeEstimator.

        Args:
            base_model_class (pyspark.ml): the model class to instantiate the TransformedOutcomeEstimator with
            base_model_params (dict): parameters and their values for the model. The model must be a regressor.
            predictors_colname (str, optional): the column names that contain the predictor variables
            propensity_model_class (, optional): the model class to instantiate the propensity model. If None, propensity will be estimated as the overall proportion of treated users in the training data.
            propensity_model_params (dict, optional): parameters and their values for the propensity model. Not used if <propensity_model_class> is None.
            treatment_colname (str, optional): the column name that contains the treatment indicators
            target_colname (str, optional): the column name that contains the target
            treatment_value (str or int, optional): the value in column <treatment_colname> that refers to the treatment group
            control_value (str or int, optional): the value in column <treatment_colname> that refers to the control group
            output_colname (str, optional): the column name for the estimator output
        """

        base_model_params["featuresCol"] = predictors_colname
        base_model_params["labelCol"] = "D"

        self.model = base_model_class(**base_model_params)
        self.propensity_estimator = PropensityEstimator(
            propensity_model_class, propensity_model_params, predictors_colname, treatment_colname, treatment_value
        )
        self.predictors_colname = predictors_colname
        self.treatment_colname = treatment_colname
        self.target_colname = target_colname
        self.control_value = control_value
        self.treatment_value = treatment_value
        self.output_colname = output_colname

    def fit(self, df_train: pyspark.sql.DataFrame, df_val: Optional[Any] = None) -> None:
        """Trains the Transformed Outcome model by first fitting a propensity model, retrieving the propensity scores for each instance,
        computing the transformed outcomes, and finally fitting a regressor on the transformed outcomes.

        Args:
            df_train (pyspark.sql.DataFrame): a dataframe containing the treatment indicators, the observed outcomes, and predictors
            df_val (pyspark.sql.DataFrame): a dataframe containing the treatment indicators, the observed outcomes, and predictors
        """

        self.propensity_estimator.fit(df_train)

        df_train = self._get_transformed_outcomes(df_train)

        self.model = self.model.fit(df_train)

    def predict(self, df: pyspark.sql.DataFrame) -> pyspark.sql.DataFrame:
        """Applies the Transformed Outcome Estimator and returns treatment effect predictions.

        Args:
            df (pyspark.sql.DataFrame): a dataframe containing predictors

        Returns:
            df (pyspark.sql.DataFrame): a dataframe containing treatment effect predictions
        """

        df_cols = df.columns

        df = self._get_transformed_outcomes(df)
        df = self.model.transform(df)
        split_udf = udf(lambda value: value, FloatType())
        pred_col = "prediction"

        df = df.withColumn(self.output_colname, split_udf(pred_col))

        df = df.select([*df_cols, self.output_colname])

        return df

    def _get_transformed_outcomes(self, df: pyspark.sql.DataFrame) -> pyspark.sql.DataFrame:
        """Applies the propensity model, computes and returns the transformed outcomes.

        Args:
            df (pyspark.sql.DataFrame): a dataframe containing the treatment indicators, the observed outcomes, and predictors

        Returns:
            df (pyspark.sql.DataFrame): a dataframe containing the transformed outcomes
        """
        df = self.propensity_estimator.predict(df)

        df = df.withColumn(
            "D",
            F.when(
                F.col(self.treatment_colname) == self.treatment_value, F.col(self.target_colname) / F.col("propensity")
            ).otherwise(-1.0 * F.col(self.target_colname) / (1 - F.col("propensity"))),
        )

        return df


class PropensityEstimator:
    """Estimates treatment propensities, either as the simple treatment proportions E[T] or by training a model for E[T | X]."""

    def __init__(
        self,
        base_model_class: Any = None,
        base_model_params: Dict = None,
        predictors_colname: str = None,
        treatment_colname: str = "treatment",
        treatment_value: int = 1,
        control_value: int = 0,
        output_colname: str = "propensity",
    ):

        """Initializes the propensity model.

        Args:
            base_model_class (pyspark.ml, optional): the model class to instantiate the propensity model. If None, propensity will be estimated as the overall proportion of treated users in the training data.
            base_model_params (dict, optional): parameters and their values for the model. Not used if <base_model_class> is None.
            predictors_colname (list of str, optional): the column names that contain the predictor variables. Not used if <base_model_class> is None.
            treatment_colname (str, optional): the column name that contains the treatment indicators
            treatment_value (str or int, optional): the value in column <treatment_colname> that refers to the treatment group
            output_colname (str, optional): the column name for the estimator output
        """

        if base_model_class is not None:
            base_model_params["featuresCol"] = predictors_colname  # type: ignore
            base_model_params["labelCol"] = treatment_colname  # type: ignore
            self.model = base_model_class(**base_model_params)
            self.predictors_colname = predictors_colname
        else:
            self.model = None
            self.propensity = 0.5

        self.treatment_colname = treatment_colname
        self.treatment_value = treatment_value
        self.control_value = control_value
        self.output_colname = output_colname

    def fit(self, df_train: pyspark.sql.DataFrame, df_val: Optional[Any] = None) -> None:
        """Fits a propensity model.
        If self.model is None, uses the proportion of treated instances in df_train to estimate E[T], independent of X.
        If self.model is instantiated, fits a full propensity model E[T | X].

        Args:
            df_train (pyspark.sql.DataFrame): a dataframe containing the treatment indicators and predictors
            df_val (pyspark.sql.DataFrame): a dataframe containing the treatment indicators and predictors
        """

        if self.model is not None:

            self.model = self.model.fit(df_train)

    def predict(self, df: pyspark.sql.DataFrame) -> pyspark.sql.DataFrame:
        """Applies the propensity model and returns treatment assignment predictions.
        If self.model is None, uses the pre-calculated treatment proportion for all instances.
        If self.model is instantiated, applies the model to get estimates E[T | X].

        Args:
            df (pyspark.sql.DataFrame): a dataframe containing predictors

        Returns:
            df (pyspark.sql.DataFrame): a dataframe containing treatment assignment predictions
        """
        df_cols = df.columns
        if self.model is None:
            treat_count = df.where(F.col(self.treatment_colname) == self.treatment_value).count()
            control_count = df.where(F.col(self.treatment_colname) == self.control_value).count()

            df = df.withColumn(
                self.output_colname,
                F.when(
                    F.col(self.treatment_colname) == self.treatment_value,
                    F.lit(treat_count / (treat_count + control_count)),
                ).otherwise(F.lit(control_count / (treat_count + control_count))),
            )

        else:
            df = self.model.transform(df)
            split_udf = udf(lambda value: value[1].item(), FloatType())
            df = df.withColumn(self.output_colname, split_udf("probability"))

        df = df.select([*df_cols, self.output_colname])

        return df


class RetrospectiveEstimator:
    """Estimates E[T | Y=1, X], which corresponds to estimating the relative treatment effect E[Y | T=1, X] / E[Y | T=0, X] in case of 50-50 treatment vs. control RCT data.

    This estimator can also used as the greedy solution for maximizing incrementality under ROI constraints, as described in Goldenberg et al. (2020) (preprint: https://drive.google.com/file/d/1E0KQ_sT09q1bpnlt9gZFFSbrx-YgGcqF/view).
    """

    def __init__(
        self,
        base_model_class: Any,
        base_model_params: Dict,
        predictors_colname: str = "features",
        treatment_colname: str = "treatment",
        target_colname: str = "outcome",
        positive_outcome_value: int = 1,
        output_colname: str = "score",
    ):

        """Initializes the Retrospective Estimator.

        Args:
            base_model_class (pyspark.ml): the model class to instantiate the Retrospective Estimator with
            base_model_params (dict): parameters and their values for the model
            predictors_colname (str, optional): the column names that contain the predictor variables
            treatment_colname (str, optional): the column name that contains the treatment indicators
            target_colname (str, optional): the column name that contains the target
            positive_outcome_value (str or int, optional): the value in column <target_colname> that refers to the positive outcome
            output_colname (str, optional): the column name for the estimator output
        """

        base_model_params["featuresCol"] = predictors_colname
        base_model_params["labelCol"] = treatment_colname

        self.model = base_model_class(**base_model_params)
        self.predictors_colname = predictors_colname
        self.treatment_colname = treatment_colname
        self.target_colname = target_colname
        self.positive_outcome_value = positive_outcome_value
        self.output_colname = output_colname

    def fit(self, df_train: pyspark.sql.DataFrame, df_val: Optional[Any] = None) -> None:
        """Trains the Retrospective Estimator E[T | Y=1, X].

        Args:
            df_train (pyspark.sql.DataFrame): a dataframe containing the treatment indicators, the observed outcomes, and predictors
            df_val (pyspark.sql.DataFrame): a dataframe containing the treatment indicators, the observed outcomes, and predictors
        """

        self.model = self.model.fit(df_train.where(F.col(self.target_colname) == self.positive_outcome_value))

    def predict(self, df: pyspark.sql.DataFrame) -> pyspark.sql.DataFrame:
        """Applies the Retrospective Estimator model and returns predictions for E[T | Y=1, X].

        Args:
            df (pyspark.sql.DataFrame): a dataframe containing predictors

        Returns:
            df (pyspark.sql.DataFrame): a dataframe containing predictions for E[T | Y=1, X]
        """

        df_cols = df.columns
        df = self.model.transform(df)

        if "probability" not in df.columns:
            split_udf = udf(lambda value: value, FloatType())
            pred_col = "prediction"
        else:
            split_udf = udf(lambda value: value[1].item(), FloatType())
            pred_col = "probability"

        df = df.withColumn(self.output_colname, split_udf(pred_col))

        df = df.select([*df_cols, self.output_colname])

        return df
