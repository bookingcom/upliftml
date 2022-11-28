import gc
from typing import Any, Dict, List, Tuple

import h2o
import numpy as np
import pandas as pd
import pyspark
from h2o.estimators.uplift_random_forest import H2OUpliftRandomForestEstimator
from pyspark.sql import functions as F

from upliftml.feature_selection.utils import (
    discretizing,
    get_feature_importance_scores_as_pdf,
)


class DivergenceFilter:
    """
    Filter feature selection method:
    Bin-based method proposed in Zhao et al. (2022), referred to as Divergence Filters.

    Note:
        * Only works with binary treatment and binary outcome variable
        * DivergenceFilter is adopted from the causalml package. See: https://github.com/uber/causalml/blob/master/causalml/feature_selection/filters.py)
    """

    def __init__(
        self,
        target_colname: str = "outcome",
        treatment_colname: str = "treatment",
        treatment_value: int = 1,
        control_value: int = 0,
        n_bins: int = 10,
        method: str = "KL",
        smooth: bool = True,
    ):
        """
        Initialize DivergenceFilter

        Args:
            target_colname (str, optional): The column name that contains the target
            treatment_colname (str, optional): The column name that contains the treatment indicators
            treatment_value (str or int, optional): The value in column <treatment_colname> that refers to the treatment group
            control_value (str or int, optional): The value in column <treatment_colname> that refers to the control group
            n_bins (int, optional): Number of bins to be used for bin-based uplift filter methods. -1 means using not discretization
            method (string, optional): The divergence method to be used to rank the features. Taking one of the following values {'KL', 'ED', 'Chi'}.
            smooth (bool, optional): Smooth label count by adding 1 in case certain labels do not occur naturally with a treatment. Prevents zero divisions.
        """
        self.target_colname = target_colname
        self.treatment_colname = treatment_colname
        self.treatment_value = treatment_value
        self.control_value = control_value
        self.n_bins = n_bins
        self.method = method
        self.smooth = smooth

    def calculate_feature_importance(self, df: pyspark.sql.DataFrame, features: List) -> Tuple:
        """
        Calculate feature importance scores

        Args:
            df (pyspark.sql.DataFrame): DataFrame containing outcome, features, and experiment group
            features (List): List of feature names, which are columns in the dataframe

        Returns:
            A tuple, containing:
            (pd.DataFrame): Dataframe containing the feature importance statistics
            (Dict): Dictionary containing for each feature the importance of its variables
        """
        feature_importance_scores = pd.DataFrame()
        features_variables_importances = {}

        # Select evaluation function
        if self.method == "KL":
            evaluation_function = self.evaluate_kl
        elif self.method == "ED":
            evaluation_function = self.evaluate_ed
        elif self.method == "Chi":
            evaluation_function = self.evaluate_chi
        else:
            print("Please select either 'KL', 'ED', or 'Chi' as method")
            return feature_importance_scores, features_variables_importances

        # Size of the Dataframe
        total_size = df.cache().count()

        # Discretizing features if their cardinality is higher than n_bins
        if self.n_bins != -1:
            df, features = discretizing(df, features, self.n_bins)

        # Get all columns
        all_columns = features.copy()
        all_columns.extend([self.treatment_colname, self.target_colname])
        df = df.select(all_columns)

        # Count the number of treatment responder, control responder, treatment non-responder, and control non-responder
        # in the entire DataFrame
        pdf_grouped_all = (
            df.groupBy(self.treatment_colname, self.target_colname).agg(F.count("*").alias("num_rows")).toPandas()
        )

        # Calculate the probabilities and sizes for treatment responder and control responder on the entire DataFrame
        summary_all = self.calculate_probabilities(pdf=pdf_grouped_all)[1]

        # Calculate the divergence between treatment responder and control responder on the entire DataFrame
        divergence_all = evaluation_function(summary_all)

        # Count the number of treatment responder, control responder, treatment non-responder, and control non-responder
        # for each feature and it's variables over the entire dataset
        df_grouped_features_rows = df.groupBy(all_columns).agg(F.count("*").alias("num_rows")).cache()

        # For each feature, sum up the counts for treatment responder, control responder, treatment non-responder, and
        # control non-responder
        for feature in features:
            feature_importance_score, features_variables_importance = self.calculate_feature_importance_per_feature(
                df=df_grouped_features_rows,
                feature_name=feature,
                total_size=total_size,
                evaluation_function=evaluation_function,
                d_all=divergence_all,
            )
            feature_importance_scores = pd.concat([feature_importance_scores, feature_importance_score])
            features_variables_importances.update(features_variables_importance)

        feature_importance_scores = feature_importance_scores.sort_values(by="score", ascending=False)
        feature_importance_scores.reset_index(drop=True, inplace=True)

        return feature_importance_scores, features_variables_importances

    def calculate_feature_importance_per_feature(
        self,
        df: pyspark.sql.DataFrame,
        feature_name: str,
        total_size: int,
        evaluation_function: Any,
        d_all: float,
    ) -> Tuple:
        """
        Calculate the feature importance score for one feature

        Args:
            df (pyspark.sql.DataFrame): DataFrame containing outcome, features, and experiment group
            feature_name (str): Name of the feature
            total_size (int): Number of all samples
            evaluation_function: The divergence method to be used to rank the features.
            d_all (float): Divergence between treatment and control responder using all the data

        Returns:
            A tuple, containing:
            (pd.DataFrame): A dataframe containing the feature importance statistics
            (Dict): Dictionary containing the features and their feature importance scores
        """
        pdf_grouped_feature_rows = (
            df.groupBy(feature_name, self.treatment_colname, self.target_colname)
            .agg(F.sum("num_rows").alias("num_rows"))
            .toPandas()
        )

        variable_keys = pdf_grouped_feature_rows[feature_name].unique().tolist()

        d_feature = 0
        d_variables_dict = {}
        for variable in variable_keys:
            summary_feature = self.calculate_probabilities(
                pdf_grouped_feature_rows.loc[pdf_grouped_feature_rows[feature_name] == variable]
            )[1]
            divergence_variable = evaluation_function(summary_feature)
            size = 0
            for x in list(summary_feature.values()):
                size += x[1]
            d_variable = divergence_variable * size / total_size
            d_variables_dict[variable] = d_variable
            d_feature += d_variable

        feature_variable_dict = {feature_name.replace("_buckets", ""): d_variables_dict}
        divergence = d_feature - d_all

        results = pd.DataFrame(
            {"feature": feature_name.replace("_buckets", ""), "score": divergence}, index=[0]
        ).reset_index(
            drop=True
        )  # type: ignore

        # Release memory
        del [pdf_grouped_feature_rows]
        gc.collect()

        return results, feature_variable_dict

    def calculate_probabilities(self, pdf: pd.DataFrame) -> Tuple:
        """
        To count the conversions and get the probabilities by treatment and control groups.

        Args:
            pdf (pd.DataFrame): Dataframe containing all the data

        Returns:
            A tuple, containing:
            (Dict): Counts of conversions by treatment groups, of the form: {'control': {0: 10, 1: 8}, 'treatment1': {0: 5, 1: 15}}
            (Dict): Probability of conversion and group size by treatment groups, of the form: {'control': [0.490, 500], 'treatment1': [0.584, 500]}
        """
        treatment_group_keys = pdf[self.treatment_colname].unique().tolist()
        y_name_keys = pdf[self.target_colname].unique().tolist()

        results = {}  # type: Dict[int, float]
        for ti in treatment_group_keys:
            results.update({ti: {}})
            for ci in y_name_keys:
                if self.smooth:
                    results[ti].update(
                        {
                            ci: (
                                pdf.loc[(pdf[self.treatment_colname] == ti) & (pdf[self.target_colname] == ci)][
                                    "num_rows"
                                ].values[0]
                            )
                            if not pdf.loc[(pdf[self.treatment_colname] == ti) & (pdf[self.target_colname] == ci)][
                                "num_rows"
                            ].empty
                            else 1
                        }
                    )
                else:
                    results[ti].update(
                        {
                            ci: (
                                pdf.loc[(pdf[self.treatment_colname] == ti) & (pdf[self.target_colname] == ci)][
                                    "num_rows"
                                ].values[0]
                            )
                        }
                    )

        # Probability of conversion and group size by treatment group
        node_summary = {}
        for treatment_group_key in results:
            n_1 = results[treatment_group_key].get(1, 0)
            n_total = results[treatment_group_key].get(1, 0) + results[treatment_group_key].get(0, 0)
            y_mean = 1.0 * n_1 / n_total
            node_summary[treatment_group_key] = [y_mean, n_total]

        return results, node_summary

    def evaluate_kl(self, node_summary: Dict) -> float:
        """
        Evaluate the divergence using Kullback-Leibler

        Args:
            node_summary (Dict): Dictionary containing the statistics for a tree node sample

        Returns:
            d_res (float): Kullback-Leibler Divergence
        """
        if self.control_value not in node_summary:
            return 0
        pc = node_summary[self.control_value][0]
        d_kl = 0.0
        for treatment_group in node_summary:
            if treatment_group != self.control_value:
                d_kl += DivergenceFilter.kl_divergence(node_summary[treatment_group][0], pc)
        return d_kl

    @staticmethod
    def kl_divergence(pk: float, qk: float) -> float:
        """
        Calculate KL Divergence for binary classification.

        Args:
            pk (float): Probability of class 1 in treatment group
            qk (float): Probability of class 1 in control group

        Returns:
            kl (float): Kullback-Leibler divergence
        """
        if qk < 0.1**6:
            qk = 0.1**6
        elif qk > 1 - 0.1**6:
            qk = 1 - 0.1**6

        if pk < 0.1**6:
            pk = 0.1**6
        elif pk > 1 - 0.1**6:
            pk = 1 - 0.1**6

        kl = pk * np.log(pk / qk) + (1 - pk) * np.log((1 - pk) / (1 - qk))
        return kl

    def evaluate_ed(self, node_summary: Dict) -> float:
        """
        Evaluate the divergence using Euclidean Distance

        Args:
            node_summary (dict): Dictionary containing the statistics for a tree node sample

        Returns:
            d_res (float): Euclidean distance
        """
        if self.control_value not in node_summary:
            return 0
        pc = node_summary[self.control_value][0]
        d_ed = 0
        for treatment_group in node_summary:
            if treatment_group != self.control_value:
                d_ed += 2 * (node_summary[treatment_group][0] - pc) ** 2
        return d_ed

    def evaluate_chi(self, node_summary: Dict) -> float:
        """
        Evaluate the divergence using Chi-Squared

        Args:
            node_summary (dict): Dictionary containing the statistics for a tree node sample

        Returns:
            d_res (float): Chi-sqaured divergence
        """
        if self.control_value not in node_summary:
            return 0
        pc = node_summary[self.control_value][0]
        d_chi = 0
        for treatment_group in node_summary:
            if treatment_group != self.control_value:
                d_chi += (node_summary[treatment_group][0] - pc) ** 2 / max(0.1**6, pc) + (
                    node_summary[treatment_group][0] - pc
                ) ** 2 / max(0.1**6, 1 - pc)
        return d_chi


class PermutationUpliftRandomForestWrapper:
    """
    Wrapper feature selection method

    Permutation with Uplift Random Forest

    In PermutationUpliftRandomForestWrapper, the feature importance is computed by measuring the difference between
    prediction errors before and after a feature is permuted.

    Note:
        * H2OUpliftRandomForestEstimator is used to make predictions
        * Only works with binary treatment and binary outcome variable
    """

    def __init__(
        self,
        durf_dict: Dict,
        factor_list: List,
        target_colname: str = "outcome",
        treatment_colname: str = "treatment",
        n_samples: int = -1,
        n_repeats: int = 3,
    ):
        """
        Initialize PermutationUpliftRandomForestWrapper

        Args:
            durf_dict (dict): Dictionary containing the hyperparameters for the H2OUpliftRandomForestEstimator
            factor_list (list): List of categorical features
            target_colname (str, optional): The column name that contains the target
            treatment_colname (str, optional): The column name that contains the treatment indicators
            n_samples (int, optional): The number of samples to be evaluated. Use -1 to use the whole dataset. Defaults to -1.
            n_repeats (int, optional): The number of repeated evaluations. Defaults to 3
        """
        self.durf_dict = durf_dict
        self.factor_list = factor_list
        self.target_colname = target_colname
        self.treatment_colname = treatment_colname
        self.n_samples = n_samples
        self.n_repeats = n_repeats

    def calculate_feature_importance(self, df: h2o.H2OFrame, features: List) -> pd.DataFrame:
        """
        Calculate feature importance scores

        Args:
            df (h2o.H2OFrame): DataFrame containing outcome, features, and experiment group
            features (list): List of feature names, which are columns in the dataframe

        Returns:
            pdf_feature_importances (pd.DataFrame): DataFrame containing the feature importance scores
        """
        for feature in self.factor_list:
            df[feature] = df[feature].asfactor()

        # Train distributed uplift random forest
        durf = H2OUpliftRandomForestEstimator(**self.durf_dict)
        durf.train(y=self.target_colname, x=features, training_frame=df)

        # Calculate feature importances
        feature_importance_scores = durf.permutation_importance(
            df, metric="qini", use_pandas=True, n_samples=self.n_samples, n_repeats=self.n_repeats, features=None
        )

        if self.n_repeats > 1:
            feature_importance_scores["Relative Importance"] = feature_importance_scores.mean(axis=1)

        # Create the final feature importance DataFrame
        feature_importance_scores = feature_importance_scores.reset_index()[["Variable", "Relative Importance"]]
        feature_importance_scores.columns = ["feature", "score"]

        return feature_importance_scores


class UpliftCurveFilter:
    """
    Filter feature selection method

    Bin-based method motivated by the uplift curve, proposed in Hu (2022)

    Note:
        * Only works with binary treatment and binary outcome variable
    """

    def __init__(
        self,
        target_colname: str = "outcome",
        treatment_colname: str = "treatment",
        treatment_value: int = 1,
        control_value: int = 0,
        n_bins: int = 10,
    ):
        """
        Initialize the UpliftCurveFilter

        Args:
            target_colname (str, optional): The column name that contains the target
            treatment_colname (str, optional): The column name that contains the treatment indicators
            treatment_value (str or int, optional): The value in column <treatment_colname> that refers to the treatment group
            control_value (str or int, optional): The value in column <treatment_colname> that refers to the control group
            n_bins (int, optional): Number of bins to be used for bin-based uplift filter methods. -1 means using not discretization
        """
        self.target_colname = target_colname
        self.treatment_colname = treatment_colname
        self.treatment_value = treatment_value
        self.control_value = control_value
        self.n_bins = n_bins

    def calculate_feature_importance(self, df: pyspark.sql.DataFrame, features: List) -> Tuple:
        """
        Calculate feature importance scores

        Args:
            df (pyspark.sql.DataFrame): DataFrame containing outcome, features, and experiment group
            features (list): List of feature names, which are columns in the dataframe

        Returns:
            A tuple, containing:
            (pd.DataFrame): Dataframe containing the feature importance statistics
            (Dict): Dictionary containing for each feature the importance of its variables
        """

        # Discretizing features if their cardinality is higher than n_bins
        if self.n_bins != -1:
            df, features = discretizing(df, features, self.n_bins)

        # Get all columns
        all_columns = features.copy()
        all_columns.extend([self.treatment_colname, self.target_colname])

        df_grouped_features_rows = (df.groupBy(all_columns).agg(F.count("*").alias("num_rows"))).cache()

        feature_importance_scores_dict = {}  # type: Dict[str, float]
        features_variables_importances = {}  # type: Dict[str, float]
        for feature in features:
            self.calculate_feature_importance_per_feature(
                df_grouped_features_rows, feature, feature_importance_scores_dict, features_variables_importances
            )

        feature_importance_scores = get_feature_importance_scores_as_pdf(feature_importance_scores_dict)
        return feature_importance_scores, features_variables_importances

    def calculate_feature_importance_per_feature(
        self,
        df: pyspark.sql.DataFrame,
        feature_name: str,
        feature_importance_scores_dict: Dict[str, float],
        features_variables_importances: Dict[str, float],
    ) -> None:
        """
        Calculate the feature importance score for one feature

        Args:
            df (pyspark.sql.DataFrame): Dataframe grouped by features, treatment, response
            feature_name (str): The name of the feature
            feature_importance_scores_dict (dict): Dictionary containing the feature importance scores
            features_variables_importances (dict): Dictionary containing each feature's variable importance
        """

        relevant_score = 0.0
        pdf_grouped_feature_rows = (
            df.groupBy(feature_name, self.treatment_colname, self.target_colname)
            .agg(F.sum("num_rows").alias("num_rows"))
            .toPandas()
        )
        variable_keys = pdf_grouped_feature_rows[feature_name].unique().tolist()
        relevant_scores_per_variables_dict = {}
        for key in variable_keys:
            score = self.calculate_uplift_curve_based_importance(pdf_grouped_feature_rows, feature_name, key)
            relevant_scores_per_variables_dict[key] = score
            relevant_score += score
        features_variables_importances.update(
            {feature_name.replace("_buckets", ""): relevant_scores_per_variables_dict}
        )  # type: ignore
        feature_importance_scores_dict[feature_name.replace("_buckets", "")] = relevant_score

    def calculate_uplift_curve_based_importance(self, pdf: pd.DataFrame, feature_name: str, key: Any) -> float:
        """
        Calculate the feature importance score for one feature using uplift curve

        Args:
            pdf (pd.DataFrame): Dataframe grouped by feature, treatment, response
            feature_name (str): Name of the feature
            key (Any): Feature's variable

        Returns:
            (float): Feature's variable importance
        """

        pdf_k = pdf.loc[pdf[feature_name] == key].copy()

        # Treatment responder
        try:
            tr = pdf_k.loc[
                (pdf_k[self.target_colname] == 1) & (pdf_k[self.treatment_colname] == self.treatment_value)
            ].iloc[0]["num_rows"]
        except IndexError:
            tr = 0
        # Control responder
        try:
            cr = pdf_k.loc[
                (pdf_k[self.target_colname] == 1) & (pdf_k[self.treatment_colname] == self.control_value)
            ].iloc[0]["num_rows"]
        except IndexError:
            cr = 0
        # Number of treatment samples
        nt = pdf_k.loc[(pdf_k[self.treatment_colname] == self.treatment_value)]["num_rows"].sum()
        # Number of control samples
        nc = pdf_k.loc[(pdf_k[self.treatment_colname] == self.control_value)]["num_rows"].sum()

        term_1 = tr / nt if nt > 0 else 0
        term_2 = cr / nc if nc > 0 else 0

        return float((term_1 - term_2) * ((nt + nc) / pdf["num_rows"].sum()))


class NetInformationValueFilter:
    """
    Filter feature selection method

    Net Information Value method proposed in Larsen (2010)

    Note:
        * Only works with binary treatment and binary outcome variable
    """

    def __init__(
        self,
        target_colname: str = "outcome",
        treatment_colname: str = "treatment",
        treatment_value: int = 1,
        control_value: int = 0,
        n_bins: int = 10,
    ):
        """
        Initialize the NetInformationValueFilter

        Args:
            target_colname (str, optional): The column name that contains the target
            treatment_colname (str, optional): The column name that contains the treatment indicators
            treatment_value (str or int, optional): The value in column <treatment_colname> that refers to the treatment group
            control_value (str or int, optional): The value in column <treatment_colname> that refers to the control group
            n_bins (int, optional): Number of bins to be used for bin-based uplift filter methods. -1 means using not discretization
        """
        self.target_colname = target_colname
        self.treatment_colname = treatment_colname
        self.treatment_value = treatment_value
        self.control_value = control_value
        self.n_bins = n_bins

    def calculate_feature_importance(self, df: pyspark.sql.DataFrame, features: List) -> Tuple:
        """
        Calculate feature importance scores

        Args:
            df (pyspark.sql.DataFrame): DataFrame containing outcome, features, and experiment group
            features (list): List of feature names, which are columns in the dataframe

        Returns:
            A tuple, containing:
            (pd.DataFrame): Dataframe containing the feature importance statistics
            (Dict): Dictionary containing for each feature the importance of its variables
        """

        if self.n_bins != -1:
            # Discretizing features if their cardinality is higher than n_bins
            df, features = discretizing(df, features, self.n_bins)

        # Get all columns
        all_columns = features.copy()
        all_columns.extend([self.treatment_colname, self.target_colname])

        df_grouped_features_rows = (df.groupBy(all_columns).agg(F.count("*").alias("num_rows"))).cache()

        pdf_grouped_groups_rows = (
            df.groupBy(self.treatment_colname, self.target_colname).agg(F.count("*").alias("num_rows")).toPandas()
        )

        feature_importance_scores_dict = {}  # type: Dict[str, float]
        features_variables_importances = {}  # type: Dict[str, float]
        for feature in features:
            self.calculate_feature_importance_per_feature(
                df_grouped_features_rows,
                pdf_grouped_groups_rows,
                feature,
                feature_importance_scores_dict,
                features_variables_importances,
            )

        feature_importance_scores = get_feature_importance_scores_as_pdf(feature_importance_scores_dict)

        return feature_importance_scores, features_variables_importances

    def calculate_feature_importance_per_feature(
        self,
        df_features: pyspark.sql.DataFrame,
        pdf_groups: pd.DataFrame,
        feature_name: str,
        feature_importance_scores_dict: Dict,
        features_variables_importances: Dict,
    ) -> None:
        """
        Calculate the feature importance score for one feature

        Args:
            df_features (pyspark.sql.DataFrame): Dataframe grouped by features, treatment, response
            pdf_groups (pd.DataFrame): Dataframe grouped by treatment, response (like a look-up table)
            feature_name (str): The name of the feature
            feature_importance_scores_dict (dict): Dictionary containing the feature importance scores
            features_variables_importances (dict): Dictionary containing each feature's variable importance
        """
        niv_score = 0.0
        pdf_grouped_feature_rows = (
            df_features.groupBy(feature_name, self.treatment_colname, self.target_colname)
            .agg(F.sum("num_rows").alias("num_rows"))
            .toPandas()
        )
        variable_keys = pdf_grouped_feature_rows[feature_name].unique().tolist()
        niv_scores_per_variables_dict = {}
        for key in variable_keys:
            score = self.calculate_net_information_value_k(pdf_grouped_feature_rows, pdf_groups, feature_name, key)
            niv_scores_per_variables_dict[key] = score
            niv_score += score
        features_variables_importances.update({feature_name.replace("_buckets", ""): niv_scores_per_variables_dict})
        niv_score = (1 / pdf_grouped_feature_rows[feature_name].nunique()) * niv_score
        feature_importance_scores_dict[feature_name.replace("_buckets", "")] = niv_score

    def calculate_net_information_value_k(
        self, pdf_feature: pd.DataFrame, pdf_groups: pd.DataFrame, feature_name: str, key: Any
    ) -> float:
        """
        Calculate net information value

        Args:
            pdf_feature (pd.DataFrame): Dataframe grouped by feature, treatment, response
            pdf_groups (pd.DataFrame): Dataframe grouped by treatment, response (like a look-up table)
            feature_name (str): The name of the feature
            key (Any): Feature's variable

        Returns:
            (float): Net Information Value
        """

        net_weight_of_evidence = self.get_net_weight_of_evidence(pdf_feature, pdf_groups, feature_name, key)

        ep_treatment_responder = self.empirical_probability(pdf_feature, pdf_groups, 1, 1, feature_name, key)

        ep_control_non_responder = self.empirical_probability(pdf_feature, pdf_groups, 0, 0, feature_name, key)

        ep_control_responder = self.empirical_probability(pdf_feature, pdf_groups, 0, 1, feature_name, key)

        ep_treatment_non_responder = self.empirical_probability(pdf_feature, pdf_groups, 1, 0, feature_name, key)

        return float(
            net_weight_of_evidence
            * (
                (ep_treatment_responder * ep_control_non_responder)
                - (ep_control_responder * ep_treatment_non_responder)
            )
        )

    def get_net_weight_of_evidence(
        self, pdf_feature: pd.DataFrame, pdf_groups: pd.DataFrame, feature_name: str, key: Any
    ) -> float:
        """
        Return the net weight of evidence

        Args:
            pdf_feature (pd.DataFrame): Dataframe grouped by feature, treatment, response
            pdf_groups (pd.DataFrame): Dataframe grouped by treatment, response (like a look-up table)
            feature_name (str): The name of the feature
            key (Any): Feature's variable

        Returns:
            (float): Net weight of evidence
        """
        return self.get_weight_of_evidence(pdf_feature, pdf_groups, 1, feature_name, key) - self.get_weight_of_evidence(
            pdf_feature, pdf_groups, 0, feature_name, key
        )

    def get_weight_of_evidence(
        self, pdf_feature: pd.DataFrame, pdf_groups: pd.DataFrame, treatment: int, feature_name: str, key: Any
    ) -> float:
        """
        Return the weight of evidence

        Args:
            pdf_feature (pd.DataFrame): Dataframe grouped by feature, treatment, response
            pdf_groups (pd.DataFrame): Dataframe grouped by treatment, response (like a look-up table)
            treatment (int): Value for the treatment column (either 0 or 1)
            feature_name (str): The name of the feature
            key (Any): Feature's variable

        Returns:
            (float): Weight of evidence
        """
        response = self.empirical_probability(pdf_feature, pdf_groups, 1, treatment, feature_name, key)
        non_response = self.empirical_probability(pdf_feature, pdf_groups, 0, treatment, feature_name, key)

        if non_response > 0:
            if response > 0:
                return np.log10(response / non_response)
            else:
                return 10
        else:
            return -10

    def empirical_probability(
        self,
        pdf_feature: pd.DataFrame,
        pdf_groups: pd.DataFrame,
        response: int,
        treatment: int,
        feature_name: str,
        key: Any,
    ) -> float:
        """
        Return the ratio between the feature's empirical treatment and y_name count and the overall empirical treatment
        and y_name count

        Args:
            pdf_feature (pd.DataFrame): Dataframe grouped by feature, treatment, response
            pdf_groups (pd.DataFrame): Dataframe grouped by treatment, response (like a look-up table)
            response (int): Value for the response column (either 0 or 1)
            treatment (int): Value for the treatment column (either 0 or 1)
            feature_name (str): The name of the feature
            key (Any): Feature's variable

        Returns:
            (float): Empirical probability
        """
        try:
            n_r_t_k = pdf_feature.loc[
                (pdf_feature[feature_name] == key)
                & (pdf_feature[self.target_colname] == response)
                & (pdf_feature[self.treatment_colname] == treatment)
            ].iloc[0]["num_rows"]
        except IndexError:
            n_r_t_k = 0

        try:
            n_r_t = pdf_groups.loc[
                (pdf_groups[self.target_colname] == response) & (pdf_groups[self.treatment_colname] == treatment)
            ].iloc[0]["num_rows"]
        except IndexError:
            n_r_t = 0

        return n_r_t_k / n_r_t if n_r_t > 0 else 0
