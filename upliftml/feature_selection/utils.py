import datetime
from typing import Dict, Tuple, Union

import pandas as pd
import pyspark
from pyspark.ml.feature import QuantileDiscretizer
from pyspark.sql import functions as F
from pyspark.sql.functions import approxCountDistinct


def get_data_between_dates(
    df: pyspark.sql.DataFrame,
    start_date: Union[datetime.datetime, int],
    end_date: Union[datetime.datetime, int],
    date_col: str = "yyyy_mm_dd",
) -> pyspark.sql.DataFrame:
    """
    Return a dataframe with samples between start_date and end_date

    Args:
        df (pyspark.sql.DataFrame): The dataframe which should be sampled between start_date and end_date
        start_date (datetime.datetime or int): Start date or int value when working with folds
        end_date (datetime.datetime or int): End date or int value when working with folds
        date_col (string, optional): Name of the date column

    Returns:
        df (pyspark.sql.DataFrame): A dataframe with samples between start_date and end_date
    """
    return df.where(F.col(date_col).between(start_date, end_date))


def linear_weighting(pdf: pd.DataFrame, window: int, feature_colname: str = "feature") -> pd.DataFrame:
    """
    Linearly weight the feature importance over time such that more recent feature importances are more important.

    Args:
        pdf (pd.DataFrame): Dataframe which contains the feature importance for each timestep
        window (int): Number of timesteps
        feature_colname (str, optional): Name of the column with the feature names

    Returns:
        pdf (pd.DataFrame): Pandas dataframe containing the weighted feature importance
    """
    pdf = pdf.copy()
    weighted_feature_importance = 0
    for i, col in enumerate(pdf.drop([feature_colname], axis=1).columns):
        i += 1
        weighted_feature_importance += (i / window) * pdf[col]
    pdf["weighted_feature_importance"] = weighted_feature_importance
    pdf = pdf.sort_values(by="weighted_feature_importance", ascending=False)

    return pdf


def min_max_normalization(pdf: pd.DataFrame, feature_colname: str = "feature") -> pd.Series:
    """
    Min-max normalization. Assumes that the first column contains the name of the features.

    Args:
        pdf (pd.DataFrame): Dataframe which should be normalized
        feature_colname (str, optional): Name of the column with the feature names

    Returns:
        pdf (pd.DataFrame): Min-max normalized pandas dataframe
    """
    return (pdf.loc[:, pdf.columns != feature_colname] - pdf.loc[:, pdf.columns != feature_colname].min()) / (  # type: ignore
        pdf.loc[:, pdf.columns != feature_colname].max() - pdf.loc[:, pdf.columns != feature_colname].min()  # type: ignore
    )


def discretizing(df: pyspark.sql.DataFrame, features: list, n_bins: int = 10) -> Tuple:
    """
    Discretize the given features using QuantileDiscretizer

    Args:
        df (pyspark.sql.DataFrame): Dataframe which should be discretized
        features (list): List of all features
        n_bins: (int, optional): Number of bins for the discretization

    Returns:
        A tuple, containing:
        (pyspark.sql.DataFrame): A dataframe with discretized columns
        (list): List of new feature names
    """
    pdf_distinct_count = df.agg(*(approxCountDistinct(F.col(c)).alias(c) for c in features)).toPandas()
    input_features = pdf_distinct_count.loc[:, (pdf_distinct_count > n_bins).any()].columns.tolist()
    output_features = [feature + "_buckets" for feature in input_features]
    qds1 = QuantileDiscretizer(inputCols=input_features, outputCols=output_features, numBuckets=n_bins)  # type: ignore
    df = qds1.fit(df).transform(df)
    features_new = [feature for feature in features if feature not in input_features]
    features_new.extend(output_features)

    return df, features_new


def get_feature_importance_scores_as_pdf(results_dict: Dict) -> pd.DataFrame:
    """
    Return the final feature importance scores as a pd.DataFrame

    Args:
        results_dict (Dict): Dictionary containing the feature importance scores for each feature

    Returns:
        pdf (pd.DataFrame): Dataframe containing the feature importance scores for each feature

    """
    pdf = pd.DataFrame(results_dict.items(), columns=["feature", "score"]).sort_values(by="score", ascending=False)  # type: ignore
    pdf.reset_index(drop=True, inplace=True)
    return pdf
