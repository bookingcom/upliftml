from typing import Any, Dict, List, Optional, Union

import h2o  # type: ignore


def _prepare_factor_cols(dfs: List[h2o.H2OFrame], colnames: List[str]) -> None:
    """Saves the original columns in the dfs as <col>_original and converts the <col> to factors.

    Args:
        dfs (list of h2o.H2OFrame): the H2O dataframes where the columns need to be changed to factors
        colnames (list of str): the names of the columns that need to be changed into factors
    """
    for col in colnames:
        for df in dfs:
            if df is not None and f"%{col}_original" not in df.columns:
                df = df.rename(columns={col: f"%{col}_original"})
                df[col] = df[f"%{col}_original"].asfactor()


def _restore_factor_cols(dfs: List[h2o.H2OFrame], colnames: List[str]) -> None:
    """Restores the original columns in the dfs from <col>_original.

    Args:
        dfs (list of h2o.H2OFrame): the H2O dataframes where the columns were changed to factors
        colnames (list of str): the names of the columns that were changed into factors
    """
    for col in colnames:
        for df in dfs:
            if df is not None and f"%{col}_original" in df.columns:
                df.pop(col)
                df = df.rename(columns={f"%{col}_original": col})


class SLearnerEstimator:
    """Estimates treatment effect by training a single model for E[Y | T, X], applying the model with T=1 and T=0 and using the difference in these estimates as the estimated treatment effect.

    The name S-learner originates from Künzel et al. (2019) (https://arxiv.org/pdf/1706.03461.pdf).
    """

    def __init__(
        self,
        base_model_class: Any,
        base_model_params: Dict,
        predictor_colnames: List[str],
        treatment_colname: str = "treatment",
        target_colname: str = "outcome",
        treatment_value: int = 1,
        control_value: int = 0,
        categorical_outcome: bool = False,
        output_colname: str = "score",
    ):
        """Initializes the S-learner.

        Args:
            base_model_class (h2o.utils.backward_compatibility.BackwardsCompatibleMeta):
              the H2O model class to instantiate the S-learner with
            base_model_params (dict): parameters and their values for the H2O model
            predictor_colnames (list of str): the column names that contain the predictor variables
            treatment_colname (str, optional): the column name that contains the treatment indicators
            target_colname (str, optional): the column name that contains the target
            treatment_value (str or int, optional): the value in column <treatment_colname> that refers to the treatment group
            control_value (str or int, optional): the value in column <treatment_colname> that refers to the control group
            categorical_outcome (bool, optional): whether to treat the outcome as categorical
            output_colname (str, optional): the column name for the estimator output
        """

        self.model = base_model_class(**base_model_params)
        self.predictor_colnames = predictor_colnames
        self.treatment_colname = treatment_colname
        self.target_colname = target_colname
        self.control_value = control_value
        self.treatment_value = treatment_value
        self.categorical_outcome = categorical_outcome
        self.output_colname = output_colname

    def fit(self, df_h2o_train: h2o.H2OFrame, df_h2o_val: Optional[h2o.H2OFrame] = None) -> None:
        """Trains the S-learner.

        Args:
            df_h2o_train (h2o.H2OFrame): a dataframe containing the treatment indicators, the observed outcomes, and predictors
            df_h2o_val (h2o.H2OFrame, optional): a dataframe containing the treatment indicators, the observed outcomes, and predictors
        """
        factor_cols = [self.treatment_colname]
        if self.categorical_outcome:
            factor_cols.append(self.target_colname)
        _prepare_factor_cols([df_h2o_train, df_h2o_val], factor_cols)

        self.model.train(
            x=self.predictor_colnames + [self.treatment_colname],
            y=self.target_colname,
            training_frame=df_h2o_train,
            validation_frame=df_h2o_val,
        )

        _restore_factor_cols([df_h2o_train, df_h2o_val], factor_cols)

    def predict(self, df_h2o: h2o.H2OFrame) -> h2o.H2OFrame:
        """Applies the S-learner and returns treatment effect predictions.

        Args:
            df_h2o (h2o.H2OFrame): a dataframe containing predictors

        Returns:
            predictions (h2o.H2OFrame): a single column containing treatment effect predictions
        """

        df_h2o["original_treatment"] = df_h2o[self.treatment_colname]

        df_h2o[self.treatment_colname] = self.control_value
        df_h2o[self.treatment_colname] = df_h2o[self.treatment_colname].asfactor()
        preds_control = self.model.predict(df_h2o)
        pred_col = "p1" if "p1" in preds_control.columns else "predict"
        preds_control = preds_control[pred_col].set_names(["pred_control"])

        df_h2o[self.treatment_colname] = self.treatment_value
        df_h2o[self.treatment_colname] = df_h2o[self.treatment_colname].asfactor()

        preds_treatment = self.model.predict(df_h2o)[pred_col].set_names(["pred_treatment"])

        df_h2o.pop(self.treatment_colname)
        df_h2o[self.treatment_colname] = df_h2o["original_treatment"]
        df_h2o.pop("original_treatment")

        return (preds_treatment - preds_control).set_names([self.output_colname])


class TLearnerEstimator:
    """Estimates treatment effect as the difference in estimates from two separate models: E[Y | T=1, X] - E[Y | T=0, X].

    The two-model approach is widely used for treatment effect estimation.
    The name T-learner originates from Künzel et al. (2019) (https://arxiv.org/pdf/1706.03461.pdf).
    """

    def __init__(
        self,
        base_model_class: Any,
        base_model_params: Dict,
        predictor_colnames: List[str],
        treatment_colname: str = "treatment",
        target_colname: str = "outcome",
        treatment_value: int = 1,
        control_value: int = 0,
        categorical_outcome: bool = False,
        output_colname: str = "score",
    ):
        """Initializes the T-learner.

        Args:
            base_model_class (h2o.utils.backward_compatibility.BackwardsCompatibleMeta): the H2O model class to
              instantiate the T-learner with
            base_model_params (dict): parameters and their values for the H2O models
            predictor_colnames (list of str): the column names that contain the predictor variables
            treatment_colname (str, optional): the column name that contains the treatment indicators
            target_colname (str, optional): the column name that contains the target
            treatment_value (str or int, optional): the value in column <treatment_colname> that refers to the treatment group
            control_value (str or int, optional): the value in column <treatment_colname> that refers to the control group
            categorical_outcome (bool, optional): whether to treat the outcome as categorical
            output_colname (str, optional): the column name for the estimator output
        """

        self.model0 = base_model_class(**base_model_params)
        self.model1 = base_model_class(**base_model_params)
        self.predictor_colnames = predictor_colnames
        self.treatment_colname = treatment_colname
        self.target_colname = target_colname
        self.control_value = control_value
        self.treatment_value = treatment_value
        self.categorical_outcome = categorical_outcome
        self.output_colname = output_colname

    def fit(self, df_h2o_train: h2o.H2OFrame, df_h2o_val: h2o.H2OFrame = None) -> None:
        """Trains the T-learner.

        Args:
            df_h2o_train (h2o.H2OFrame): a dataframe containing the treatment indicators, the observed outcomes, and predictors
            df_h2o_val (h2o.H2OFrame, optional): a dataframe containing the treatment indicators, the observed outcomes, and predictors
        """

        factor_cols = []
        if self.categorical_outcome:
            factor_cols.append(self.target_colname)
        _prepare_factor_cols([df_h2o_train, df_h2o_val], factor_cols)

        df_h2o_val_filtered = None
        if df_h2o_val is not None:
            df_h2o_val_filtered = df_h2o_val[df_h2o_val[self.treatment_colname] == self.treatment_value]

        self.model1.train(
            x=self.predictor_colnames,
            y=self.target_colname,
            training_frame=df_h2o_train[df_h2o_train[self.treatment_colname] == self.treatment_value],
            validation_frame=df_h2o_val_filtered,
        )

        if df_h2o_val is not None:
            df_h2o_val_filtered = df_h2o_val[df_h2o_val[self.treatment_colname] == self.control_value]

        self.model0.train(
            x=self.predictor_colnames,
            y=self.target_colname,
            training_frame=df_h2o_train[df_h2o_train[self.treatment_colname] == self.control_value],
            validation_frame=df_h2o_val_filtered,
        )

        _restore_factor_cols([df_h2o_train, df_h2o_val], factor_cols)

    def predict(self, df_h2o: h2o.H2OFrame) -> h2o.H2OFrame:
        """Applies the T-learner and returns treatment effect predictions.

        Args:
            df_h2o (h2o.H2OFrame): a dataframe containing predictors

        Returns:
            predictions (h2o.H2OFrame): a single column containing treatment effect predictions
        """

        preds_control = self.model0.predict(df_h2o)
        pred_col = "p1" if "p1" in preds_control.columns else "predict"
        preds_control = preds_control[pred_col].set_names(["pred_control"])
        preds_treatment = self.model1.predict(df_h2o)[pred_col].set_names(["pred_treatment"])
        return (preds_treatment - preds_control).set_names([self.output_colname])


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
        predictor_colnames_1: List[str],
        base_model_class_2: Any,
        base_model_params_2: Dict,
        predictor_colnames_2: List[str],
        treatment_colname: str = "treatment",
        target_colname: str = "outcome",
        treatment_value: int = 1,
        control_value: int = 0,
        categorical_outcome: bool = False,
        output_colname: str = "score",
    ):
        """Initializes the X-learner.

        Args:
            base_model_class_1 (h2o.utils.backward_compatibility.BackwardsCompatibleMeta): the H2O model class to instantiate the first stage learners
            base_model_params_1 (dict): parameters and their values for the first stage H2O models
            predictor_colnames_1 (list of str): the column names that contain the predictor variables for the first stage models
            base_model_class_2 (h2o.utils.backward_compatibility.BackwardsCompatibleMeta): the H2O model class to instantiate the second stage learners
            base_model_params_2 (dict): parameters and their values for the second stage H2O models. The models must be regressors.
            predictor_colnames_2 (list of str): the column names that contain the predictor variables for the second stage models
            treatment_colname (str, optional): the column name that contains the treatment indicators
            target_colname (str, optional): the column name that contains the target
            treatment_value (str or int, optional): the value in column <treatment_colname> that refers to the treatment group
            control_value (str or int, optional): the value in column <treatment_colname> that refers to the control group
            categorical_outcome (bool, optional): whether to treat the outcome as categorical
            output_colname (str, optional): the column name for the estimator output
        """

        self.first_learner = TLearnerEstimator(
            base_model_class_1,
            base_model_params_1,
            predictor_colnames_1,
            treatment_colname,
            target_colname,
            treatment_value,
            control_value,
        )

        self.second_learner = TLearnerEstimator(
            base_model_class_2,
            base_model_params_2,
            predictor_colnames_2,
            treatment_colname,
            "D",
            treatment_value,
            control_value,
        )

        self.treatment_colname = treatment_colname
        self.target_colname = target_colname
        self.control_value = control_value
        self.treatment_value = treatment_value
        self.categorical_outcome = categorical_outcome
        self.output_colname = output_colname

    def fit(self, df_h2o_train: h2o.H2OFrame, df_h2o_val: Optional[h2o.H2OFrame] = None) -> None:
        """Trains the X-learner.

        Args:
            df_h2o_train (h2o.H2OFrame): a dataframe containing the treatment indicators, the observed outcomes, and predictors
            df_h2o_val (h2o.H2OFrame, optional): a dataframe containing the treatment indicators, the observed outcomes, and predictors
        """

        factor_cols = []
        if self.categorical_outcome:
            factor_cols.append(self.target_colname)
        _prepare_factor_cols([df_h2o_train, df_h2o_val], factor_cols)

        self.first_learner.fit(df_h2o_train, df_h2o_val)

        _restore_factor_cols([df_h2o_train, df_h2o_val], factor_cols)

        df_h2o_train["D"] = self._get_second_learner_targets(df_h2o_train)

        if df_h2o_val is not None and "D" not in df_h2o_val.columns:
            df_h2o_val["D"] = self._get_second_learner_targets(df_h2o_val)

        self.second_learner.fit(df_h2o_train, df_h2o_val)

        df_h2o_train.pop("D")
        if df_h2o_val is not None and "D" in df_h2o_val.columns:
            df_h2o_val.pop("D")

    def predict(self, df_h2o: h2o.H2OFrame) -> h2o.H2OFrame:
        """Applies the X-learner and returns treatment effect predictions.

        Args:
            df_h2o (h2o.H2OFrame): a dataframe containing predictors

        Returns:
            predictions (h2o.H2OFrame): a single column containing treatment effect predictions
        """

        preds_control = self.second_learner.model0.predict(df_h2o)["predict"].set_names(["pred_control"])
        preds_treatment = self.second_learner.model1.predict(df_h2o)["predict"].set_names(["pred_treatment"])
        return (0.5 * (preds_treatment + preds_control)).set_names([self.output_colname])

    def _get_second_learner_targets(self, df_h2o: h2o.H2OFrame) -> h2o.H2OFrame:
        """Applies the first stage models and returns the targets for the second stage model.

        Args:
            df_h2o (h2o.H2OFrame): a dataframe containing the treatment indicators, the observed outcomes, and predictors

        Returns:
            targets (h2o.H2OFrame): a single column containing the second stage targets for the X-learner
        """

        preds0 = self.first_learner.model0.predict(df_h2o)
        pred_col = "p1" if "p1" in preds0.columns else "predict"
        preds0 = preds0[pred_col]
        preds1 = self.first_learner.model1.predict(df_h2o)[pred_col]

        return (df_h2o[self.treatment_colname] == self.treatment_value).ifelse(
            df_h2o[self.target_colname] - preds0, preds1 - df_h2o[self.target_colname]
        )


class RLearnerEstimator:
    """Estimates treatment effect in two stages:
    1. Using cross-fitting, on the training set train a marginal target estimator to get scores Y_hat and a propensity estimator to get scores T_hat, calculate the residuals on the validation set.
    2. Train a final estimator on the residuals: tau = E[(Y- Y_hat) / (T - T_hat) | X] using (T - T_hat)^2 as weights.

    R-learner was proposed in Nie and Wager (2019) (https://arxiv.org/abs/1712.04912).
    """

    def __init__(
        self,
        target_model_class: Any,
        target_model_params: Dict,
        final_model_class: Any,
        final_model_params: Dict,
        predictor_colnames: List[str],
        propensity_model_class: Any = None,
        propensity_model_params: Any = None,
        treatment_colname: str = "treatment",
        target_colname: str = "outcome",
        treatment_value: int = 1,
        control_value: int = 0,
        categorical_outcome: bool = False,
        fold_colname: str = None,
        n_folds: int = None,
        output_colname: str = "score",
    ):
        """Initializes the R-learner.

        Args:
            target_model_class (h2o.utils.backward_compatibility.BackwardsCompatibleMeta): the H2O model class to instantiate the marginal target estimator
            target_model_params (dict): parameters and their values for the marginal target estimator H2O model
            final_model_class (h2o.utils.backward_compatibility.BackwardsCompatibleMeta): the H2O model class to instantiate the final (second stage) model
            final_model_params (dict): parameters and their values for the final (second stage) H2O model. The model must be a regressor.
            predictor_colnames (list of str): the column names that contain the predictor variables (used in all of the submodels)
            propensity_model_class (h2o.utils.backward_compatibility.BackwardsCompatibleMeta, optional): the H2O model class to instantiate the propensity model. If None, propensity will be estimated as the overall proportion of treated users in the training data.
            propensity_model_params (dict, optional): parameters and their values for the propensity H2O model. Not used if <propensity_model_class> is None.
            treatment_colname (str, optional): the column name that contains the treatment indicators
            target_colname (str, optional): the column name that contains the target
            treatment_value (str or int, optional): the value in column <treatment_colname> that refers to the treatment group
            control_value (str or int, optional): the value in column <treatment_colname> that refers to the control group
            categorical_outcome (bool, optional): whether to treat the outcome as categorical
            fold_colname (str, optional): name of the column that contains the fold assignments. If neither fold_colname nor n_folds is specified, a simple cross-fitting will be performed by splitting the data into two.
            n_folds (str, optional): The number of folds to split the data into for cross-fitting purposes. If neither fold_colname nor n_folds is specified, a simple cross-fitting will be performed by splitting the data into two.
            output_colname (str, optional): the column name for the estimator output
        """

        self.target_model_class = target_model_class
        self.target_model_params = target_model_params
        self.propensity_model_class = propensity_model_class
        self.propensity_model_params = propensity_model_params

        self.model_final = final_model_class(**final_model_params)
        self.predictor_colnames = predictor_colnames
        self.treatment_colname = treatment_colname
        self.target_colname = target_colname
        self.control_value = control_value
        self.treatment_value = treatment_value
        self.categorical_outcome = categorical_outcome
        self.fold_colname = fold_colname
        self.n_folds = n_folds
        self.output_colname = output_colname

        if self.fold_colname is None and self.n_folds is None:
            self.n_folds = 2

    def fit(self, df_h2o_train: h2o.H2OFrame, df_h2o_val: Optional[h2o.H2OFrame] = None) -> None:
        """Trains the R-learner.

        Args:
            df_h2o_train (h2o.H2OFrame): a dataframe containing the treatment indicators, the observed outcomes, and predictors
            df_h2o_val (h2o.H2OFrame, optional): a dataframe containing the treatment indicators, the observed outcomes, and predictors
        """

        # make copies of the original dfs to avoid modifying them
        df_h2o_train_copy = h2o.deep_copy(df_h2o_train, "df_h2o_train_copy")
        if df_h2o_val is not None:
            df_h2o_val_copy = h2o.deep_copy(df_h2o_val, "df_h2o_val_copy")

        # assign folds if needed
        if self.fold_colname is None:
            df_h2o_train_copy["fold"] = df_h2o_train_copy.kfold_column(n_folds=self.n_folds, seed=42)
            self.fold_colname = "fold"

        # if validation set is provided, add it as an additional fold to the training set
        if df_h2o_val is not None:
            df_h2o_val_copy[self.fold_colname] = df_h2o_train_copy[self.fold_colname].max() + 1
            df_h2o_train_copy = df_h2o_train_copy.rbind(df_h2o_val_copy)

        # cross-validation cross-fitting
        final_train = None
        for fold in df_h2o_train_copy[self.fold_colname].unique().as_data_frame()["C1"]:
            current_train = df_h2o_train_copy[df_h2o_train_copy[self.fold_colname] != fold]
            current_test = df_h2o_train_copy[df_h2o_train_copy[self.fold_colname] == fold]

            current_test = self._fit_nuisance_models_and_get_residual_targets(current_train, current_test)

            if final_train is None:
                final_train = current_test
            else:
                final_train = final_train.rbind(current_test)

        # fit the final model on cross-fitted residuals
        self.model_final.train(
            x=self.predictor_colnames, y="residual_target", training_frame=final_train, weights_column="weight"
        )

    def predict(self, df_h2o: h2o.H2OFrame) -> h2o.H2OFrame:
        """Applies the R-learner and returns treatment effect predictions.

        Args:
            df_h2o (h2o.H2OFrame): a dataframe containing predictors

        Returns:
            predictions (h2o.H2OFrame): a single column containing treatment effect predictions
        """

        return self.model_final.predict(df_h2o)["predict"].set_names([self.output_colname])

    def _fit_nuisance_models_and_get_residual_targets(
        self, df_h2o_train: h2o.H2OFrame, df_h2o_val: h2o.H2OFrame
    ) -> h2o.H2OFrame:
        """Trains the first stage models on train set, applies them to the val set, and returns the val set with residual targets and weights.

        Args:
            df_h2o_train (h2o.H2OFrame): a dataframe containing the treatment indicators, the observed outcomes, and predictors
            df_h2o_val (h2o.H2OFrame): a dataframe containing the treatment indicators, the observed outcomes, and predictors

        Returns:
            df_h2o_val (h2o.H2OFrame): a dataframe containing the treatment indicators, the observed outcomes, and predictors, residual targets, and weights
        """

        factor_cols = [self.treatment_colname]
        if self.categorical_outcome:
            factor_cols.append(self.target_colname)
        _prepare_factor_cols([df_h2o_train], factor_cols)

        # fit nuisance models on train chunk
        model_y = self.target_model_class(**self.target_model_params)
        propensity_model = PropensityEstimator(
            self.propensity_model_class,
            self.propensity_model_params,
            self.predictor_colnames,
            self.treatment_colname,
            self.treatment_value,
        )

        model_y.train(x=self.predictor_colnames, y=self.target_colname, training_frame=df_h2o_train)

        propensity_model.fit(df_h2o_train)

        _restore_factor_cols([df_h2o_train], factor_cols)

        # get residuals for test chunk
        preds_y = model_y.predict(df_h2o_val)
        preds_t = propensity_model.predict(df_h2o_val)
        pred_col_y = "p1" if "p1" in preds_y.columns else "predict"

        residual_y = df_h2o_val[self.target_colname] - preds_y[pred_col_y]
        residual_t = df_h2o_val[self.treatment_colname] - preds_t["propensity"]

        df_h2o_val["residual_target"] = residual_y / residual_t
        df_h2o_val["weight"] = residual_t ** 2

        return df_h2o_val


class TransformedOutcomeEstimator:
    """Estimates treatment effect by transforming the outcome, such that the expectation of the transformed outcome corresponds to the treatment effect.
    This transformation results in a regression problem and is, thus, slightly different from the CVTEstimator, which results in a classification problem.

    The Transformed Outcome technique was proposed in Athey and Imbens (2015) (https://pdfs.semanticscholar.org/86ce/004214845a1683d59b64c4363a067d342cac.pdf).
    """

    def __init__(
        self,
        base_model_class: Any,
        base_model_params: Dict,
        predictor_colnames: List[str],
        propensity_model_class: Any = None,
        propensity_model_params: Optional[Dict] = None,
        treatment_colname: str = "treatment",
        target_colname: str = "outcome",
        treatment_value: int = 1,
        control_value: int = 0,
        output_colname: str = "score",
    ):
        """Initializes the TransformedOutcomeEstimator.

        Args:
            base_model_class (h2o.utils.backward_compatibility.BackwardsCompatibleMeta): the H2O model class to instantiate the TransformedOutcomeEstimator with
            base_model_params (dict): parameters and their values for the H2O model. The model must be a regressor.
            predictor_colnames (list of str): the column names that contain the predictor variables
            propensity_model_class (h2o.utils.backward_compatibility.BackwardsCompatibleMeta, optional): the H2O model class to instantiate the propensity model. If None, propensity will be estimated as the overall proportion of treated users in the training data.
            propensity_model_params (dict, optional): parameters and their values for the propensity H2O model. Not used if <propensity_model_class> is None.
            treatment_colname (str, optional): the column name that contains the treatment indicators
            target_colname (str, optional): the column name that contains the target
            treatment_value (str or int, optional): the value in column <treatment_colname> that refers to the treatment group
            control_value (str or int, optional): the value in column <treatment_colname> that refers to the control group
            output_colname (str, optional): the column name for the estimator output
        """

        self.model = base_model_class(**base_model_params)
        self.propensity_model = PropensityEstimator(
            propensity_model_class, propensity_model_params, predictor_colnames, treatment_colname, treatment_value
        )
        self.predictor_colnames = predictor_colnames
        self.treatment_colname = treatment_colname
        self.target_colname = target_colname
        self.control_value = control_value
        self.treatment_value = treatment_value
        self.output_colname = output_colname

    def fit(self, df_h2o_train: h2o.H2OFrame, df_h2o_val: Optional[h2o.H2OFrame] = None) -> None:
        """Trains the Transformed Outcome model by first fitting a propensity model, retrieving the propensity scores for each instance,
        computing the transformed outcomes, and finally fitting a regressor on the transformed outcomes.

        Args:
            df_h2o_train (h2o.H2OFrame): a dataframe containing the treatment indicators, the observed outcomes, and predictors
            df_h2o_val (h2o.H2OFrame, optional): a dataframe containing the treatment indicators, the observed outcomes, and predictors
        """
        self.propensity_model.fit(df_h2o_train, df_h2o_val)
        df_h2o_train["transformed_outcome"] = self._get_transformed_outcomes(df_h2o_train)

        if df_h2o_val is not None and "transformed_outcome" not in df_h2o_val.columns:
            df_h2o_val["transformed_outcome"] = self._get_transformed_outcomes(df_h2o_val)

        self.model.train(
            x=self.predictor_colnames, y="transformed_outcome", training_frame=df_h2o_train, validation_frame=df_h2o_val
        )

        df_h2o_train.pop("transformed_outcome")
        if df_h2o_val is not None and "transformed_outcome" in df_h2o_val.columns:
            df_h2o_val.refresh()  # without refreshing sometimes the 'transformed_outcome' column is not found
            df_h2o_val.pop("transformed_outcome")

    def predict(self, df_h2o: h2o.H2OFrame) -> h2o.H2OFrame:
        """Applies the Transformed Outcome Estimator and returns treatment effect predictions.

        Args:
            df_h2o (h2o.H2OFrame): a dataframe containing predictors

        Returns:
            predictions (h2o.H2OFrame): a single column containing treatment effect predictions
        """

        return (self.model.predict(df_h2o)["predict"]).set_names([self.output_colname])

    def _get_transformed_outcomes(self, df_h2o: h2o.H2OFrame) -> h2o.H2OFrame:
        """Applies the propensity model, computes and returns the transformed outcomes.

        Args:
            df_h2o (h2o.H2OFrame): a dataframe containing the treatment indicators, the observed outcomes, and predictors

        Returns:
            targets (h2o.H2OFrame): a single column containing the transformed outcomes
        """
        propensities = self.propensity_model.predict(df_h2o)
        return (df_h2o[self.treatment_colname] == self.treatment_value).ifelse(
            df_h2o[self.target_colname] / propensities, -1.0 * df_h2o[self.target_colname] / (1 - propensities)
        )


class PropensityEstimator:
    """Estimates treatment propensities, either as the simple treatment proportions E[T] or by training a model for E[T | X]."""

    def __init__(
        self,
        base_model_class: Any = None,
        base_model_params: Dict = None,
        predictor_colnames: List[str] = None,
        treatment_colname: str = "treatment",
        treatment_value: Union[int, str] = 1,
        output_colname: str = "propensity",
    ):
        """Initializes the propensity model.

        Args:
            base_model_class (h2o.utils.backward_compatibility.BackwardsCompatibleMeta, optional): the H2O model class to instantiate the propensity model. If None, propensity will be estimated as the overall proportion of treated users in the training data.
            base_model_params (dict, optional): parameters and their values for the H2O model. Not used if <base_model_class> is None.
            predictor_colnames (list of str, optional): the column names that contain the predictor variables. Not used if <base_model_class> is None.
            treatment_colname (str, optional): the column name that contains the treatment indicators
            treatment_value (str or int, optional): the value in column <treatment_colname> that refers to the treatment group
            output_colname (str, optional): the column name for the estimator output
        """

        if base_model_class is not None:
            self.model = base_model_class(**base_model_params)
            self.predictor_colnames = predictor_colnames
        else:
            self.model = None
            self.propensity = 0.5

        self.treatment_colname = treatment_colname
        self.treatment_value = treatment_value
        self.output_colname = output_colname

    def fit(self, df_h2o_train: h2o.H2OFrame, df_h2o_val: Optional[h2o.H2OFrame] = None) -> None:
        """Fits a propensity model.
        If self.model is None, uses the proportion of treated instances in df_h2o_train to estimate E[T], independent of X.
        If self.model is instantiated, fits a full propensity model E[T | X].

        Args:
            df_h2o_train (h2o.H2OFrame): a dataframe containing the treatment indicators and predictors
            df_h2o_val (h2o.H2OFrame, optional): a dataframe containing the treatment indicators and predictors
        """

        if self.model is None:
            counts = df_h2o_train[self.treatment_colname].table().as_data_frame()
            counts["propensity"] = counts["Count"] / counts["Count"].sum()
            self.propensity = counts[counts[self.treatment_colname] == self.treatment_value]["propensity"].squeeze()
        else:
            factor_cols = [self.treatment_colname]
            _prepare_factor_cols([df_h2o_train, df_h2o_val], factor_cols)

            self.model.train(
                x=self.predictor_colnames,
                y=self.treatment_colname,
                training_frame=df_h2o_train,
                validation_frame=df_h2o_val,
            )

            _restore_factor_cols([df_h2o_train, df_h2o_val], factor_cols)

    def predict(self, df_h2o: h2o.H2OFrame) -> h2o.H2OFrame:
        """Applies the propensity model and returns treatment assignment predictions.
        If self.model is None, uses the pre-calculated treatment proportion for all instances.
        If self.model is instantiated, applies the model to get estimates E[T | X].

        Args:
            df_h2o (h2o.H2OFrame): a dataframe containing predictors

        Returns:
            propensities (h2o.H2OFrame): a single column containing treatment assignment predictions
        """
        if self.model is None:
            return h2o.H2OFrame([self.propensity] * len(df_h2o)).set_names([self.output_colname])
        else:
            return (self.model.predict(df_h2o)["p1"]).set_names([self.output_colname])


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
        predictor_colnames: List[str],
        treatment_colname: str = "treatment",
        target_colname: str = "outcome",
        output_colname: str = "score",
    ):
        """Initializes the CVTEstimator.

        Args:
            base_model_class (h2o.utils.backward_compatibility.BackwardsCompatibleMeta): the H2O model class to instantiate the CVTEstimator with
            base_model_params (dict): parameters and their values for the H2O model
            predictor_colnames (list of str): the column names that contain the predictor variables
            treatment_colname (str, optional): the column name that contains the treatment indicators
            target_colname (str, optional): the column name that contains the target
            output_colname (str, optional): the column name for the estimator output
        """

        self.model = base_model_class(**base_model_params)
        self.predictor_colnames = predictor_colnames
        self.treatment_colname = treatment_colname
        self.target_colname = target_colname
        self.output_colname = output_colname

    def fit(self, df_h2o_train: h2o.H2OFrame, df_h2o_val: Optional[h2o.H2OFrame] = None) -> None:
        """Trains the CVT model by transforming the target variable and fitting a classifier on the transformed targets.

        Args:
            df_h2o_train (h2o.H2OFrame): a dataframe containing the treatment indicators, the observed outcomes, and predictors
            df_h2o_val (h2o.H2OFrame, optional): a dataframe containing the treatment indicators, the observed outcomes, and predictors
        """

        df_h2o_train["cvt_label"] = df_h2o_train[self.treatment_colname] == df_h2o_train[self.target_colname]
        df_h2o_train["cvt_label"] = df_h2o_train["cvt_label"].asfactor()

        if df_h2o_val is not None and "cvt_label" not in df_h2o_val.columns:
            df_h2o_val["cvt_label"] = df_h2o_val[self.treatment_colname] == df_h2o_val[self.target_colname]
            df_h2o_val["cvt_label"] = df_h2o_val["cvt_label"].asfactor()

        self.model.train(
            x=self.predictor_colnames, y="cvt_label", training_frame=df_h2o_train, validation_frame=df_h2o_val
        )

        df_h2o_train.pop("cvt_label")
        if df_h2o_val is not None and "cvt_label" in df_h2o_val.columns:
            df_h2o_val.pop("cvt_label")

    def predict(self, df_h2o: h2o.H2OFrame) -> h2o.H2OFrame:
        """Applies the CVT model and returns treatment effect predictions.

        Args:
            df_h2o (h2o.H2OFrame): a dataframe containing predictors

        Returns:
            predictions (h2o.H2OFrame): a single column containing treatment effect predictions
        """
        return (2 * self.model.predict(df_h2o)["p1"] - 1).set_names([self.output_colname])


class RetrospectiveEstimator:
    """Estimates E[T | Y=1, X], which corresponds to estimating the relative treatment effect E[Y | T=1, X] / E[Y | T=0, X] in case of 50-50 treatment vs. control RCT data.

    This estimator can also used as the greedy solution for maximizing incrementality under ROI constraints, as described in Goldenberg et al. (2020) (preprint: https://drive.google.com/file/d/1E0KQ_sT09q1bpnlt9gZFFSbrx-YgGcqF/view).
    """

    def __init__(
        self,
        base_model_class: Any,
        base_model_params: Dict,
        predictor_colnames: List[str],
        treatment_colname: str = "treatment",
        target_colname: str = "outcome",
        positive_outcome_value: Union[str, int] = 1,
        output_colname: str = "score",
    ):
        """Initializes the Retrospective Estimator.

        Args:
            base_model_class (h2o.utils.backward_compatibility.BackwardsCompatibleMeta): the H2O model class to instantiate the Retrospective Estimator with
            base_model_params (dict): parameters and their values for the H2O model
            predictor_colnames (list of str): the column names that contain the predictor variables
            treatment_colname (str, optional): the column name that contains the treatment indicators
            target_colname (str, optional): the column name that contains the target
            positive_outcome_value (str or int, optional): the value in column <target_colname> that refers to the positive outcome
            output_colname (str, optional): the column name for the estimator output
        """

        self.model = base_model_class(**base_model_params)
        self.predictor_colnames = predictor_colnames
        self.treatment_colname = treatment_colname
        self.target_colname = target_colname
        self.positive_outcome_value = positive_outcome_value
        self.output_colname = output_colname

    def fit(self, df_h2o_train: h2o.H2OFrame, df_h2o_val: Optional[h2o.H2OFrame] = None) -> None:
        """Trains the Retrospective Estimator E[T | Y=1, X].

        Args:
            df_h2o_train (h2o.H2OFrame): a dataframe containing the treatment indicators, the observed outcomes, and predictors
            df_h2o_val (h2o.H2OFrame, optional): a dataframe containing the treatment indicators, the observed outcomes, and predictors
        """

        factor_cols = [self.treatment_colname]
        _prepare_factor_cols([df_h2o_train, df_h2o_val], factor_cols)

        df_h2o_val_filtered = None
        if df_h2o_val is not None:
            df_h2o_val_filtered = df_h2o_val[df_h2o_val[self.target_colname] == self.positive_outcome_value]

        self.model.train(
            x=self.predictor_colnames,
            y=self.treatment_colname,
            training_frame=df_h2o_train[df_h2o_train[self.target_colname] == self.positive_outcome_value],
            validation_frame=df_h2o_val_filtered,
        )

        _restore_factor_cols([df_h2o_train, df_h2o_val], factor_cols)

    def predict(self, df_h2o: h2o.H2OFrame) -> h2o.H2OFrame:
        """Applies the Retrospective Estimator model and returns predictions for E[T | Y=1, X].

        Args:
            df_h2o (h2o.H2OFrame): a dataframe containing predictors

        Returns:
            predictions (h2o.H2OFrame): a single column containing predictions for E[T | Y=1, X]
        """
        return self.model.predict(df_h2o)["p1"].set_names([self.output_colname])
