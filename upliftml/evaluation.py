import itertools
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyspark.sql.functions as F
import pyspark.sql.types as T
import seaborn as sns  # type: ignore
from pyspark.ml.feature import QuantileDiscretizer
from pyspark.sql import Column, DataFrame, Window
from sklearn.metrics import auc  # type: ignore


def _generate_bootstrap_rows(df: DataFrame, n_bootstraps: int = 100) -> DataFrame:
    """Adds rows with samples from the Poisson distribution to the dataframe.
    The number of rows will be multiplied by n_bootstraps and a column 'n_draws'
    will reflect the number of times a given row was selected in a given bootstrap.
    The 0th bootstrap will contain the original data with n_draws set to 1 for each row.
    For an explanation on Poisson bootstrapping,
    see http://www.unofficialgoogledatascience.com/2015/08/an-introduction-to-poisson-bootstrap26.html.

    Args:
        df (pyspark.sql.DataFrame): a Spark dataframe
        n_bootstraps (int, optional): the number of bootstraps to perform. Only has an effect if bootstrap=True

    Returns:
        (pyspark.sql.DataFrame): a Spark dataframe containing the original data and the bootstrap samples
    """

    @F.udf(T.ArrayType(T.IntegerType()))
    def _poisson_array(nb: int) -> List[int]:
        return [int(p) for p in np.random.poisson(1, nb)]  # type: ignore

    _poisson_array = _poisson_array.asNondeterministic()  # type: ignore

    cols = df.columns

    df = df.withColumn("poissonVector", _poisson_array(F.lit(n_bootstraps)))
    return df.select(*cols, F.posexplode("poissonVector").alias("bootstrap", "n_draws")).withColumn(
        "n_draws", F.when(F.col("bootstrap") == 0, F.lit(1)).otherwise(F.col("n_draws"))
    )


def _estimate_metrics(
    df: DataFrame,
    metric_type: str = "ate",
    target_colname: str = "outcome",
    benefit_colname: str = "revenue",
    cost_colname: str = "cost",
    treatment_colname: str = "treatment",
    treatment_value: Union[int, str] = 1,
    control_value: Union[int, str] = 0,
    bucket_colname: Optional[str] = None,
    cumulative: bool = False,
    bootstrap: bool = False,
    n_bootstraps: int = 100,
) -> DataFrame:
    """Estimates metrics related to either average treatment effect, ROI, or incremental ROI.
    Runs bootstrapping if requested.

    Args:
        df (pyspark.sql.DataFrame): a Spark dataframe
        metric_type (str, optional): the type of the metrics to estimate: 'ate', 'roi', 'iroi', or 'target_rate'.
        benefit_colname (str, optional): the column name in df that contains the benefit.
            Used if metric_type is either 'roi' or 'iroi'.
        cost_colname (str, optional): the column name in df that contains the cost.
            Used if metric_type is either 'roi' or 'iroi'.
        treatment_colname (str, optional): the column name in df that contains the treatment indicators.
            Used if metric_type is either 'ate'.
        treatment_value (str or int, optional): the value in column <treatment_colname>
            that refers to the treatment group
        control_value (str or int, optional): the value in column <treatment_colname> that refers to the control group
        bucket_colname (str, optional): the column name that contains the bucket assignments.
            If None, each row is assumed to represent the whole population
        cumulative (bool, optional): if True, will calculate the metrics in a cumulative manner,
            with rows ordered from the highest-valued to lowest-valued bucket.
        bootstrap (bool, optional): if True, will perform bootstrapping and return confidence intervals
        n_bootstraps (int, optional): the number of bootstraps to perform. Only has an effect if bootstrap=True

    Returns:
        (pyspark.sql.DataFrame): a Spark dataframe containing point estimates and confidence
            intervals (aggregated over bootstraps)
    """

    def cnt_cond(cond: Column) -> Column:
        return F.sum(F.when(cond, multiplier).otherwise(0))

    def sum_cond(cond: Column, colname: str) -> Column:
        return F.sum(F.when(cond, F.col(colname) * multiplier).otherwise(0))

    group_cols = [] if bucket_colname is None else [bucket_colname]
    if "threshold" in df.columns:
        group_cols.append("threshold")
    if bootstrap:
        # perform bootstrapping
        df = _generate_bootstrap_rows(df, n_bootstraps=n_bootstraps)

        multiplier = F.col("n_draws")
        group_cols += ["bootstrap"]

    else:
        multiplier = F.lit(1)

    # Define (weighted) count and sum conditions

    is_treatment = F.col(treatment_colname) == treatment_value
    is_control = F.col(treatment_colname) == control_value

    # Calculate necessary aggregations that are needed for calculating the final the metrics of interest
    agg_exps = [F.sum(multiplier).alias("count")]
    if metric_type == "ate":
        agg_exps += [
            cnt_cond(is_treatment).alias("treatment_count"),
            cnt_cond(is_control).alias("control_count"),
            sum_cond(is_treatment, target_colname).alias("treatment_target"),
            sum_cond(is_control, target_colname).alias("control_target"),
        ]

    elif metric_type == "roi":
        agg_exps += [
            F.sum(F.col(benefit_colname) * multiplier).alias("tot_benefit"),
            F.sum(F.col(cost_colname) * multiplier).alias("tot_cost"),
        ]

    elif metric_type == "iroi":
        agg_exps += [
            cnt_cond(is_treatment).alias("treatment_count"),
            cnt_cond(is_control).alias("control_count"),
            sum_cond(is_treatment, benefit_colname).alias("treatment_benefit"),
            sum_cond(is_control, benefit_colname).alias("control_benefit"),
            sum_cond(is_treatment, cost_colname).alias("treatment_cost"),
            sum_cond(is_control, cost_colname).alias("control_cost"),
        ]

    elif metric_type == "target_rate":
        agg_exps += [F.sum(F.col(target_colname) * multiplier).alias("tot_target")]

    df_counts = df.groupBy(group_cols).agg(*agg_exps)

    # Add fraction of population
    df_counts = _add_fraction_of_population(df_counts, bucket_colname=bucket_colname)

    # For some plots, we need to make the metrics cumulative
    if cumulative:
        metric_cols = [col for col in df_counts.columns if col not in group_cols]
        df_counts = _add_cumulative_metrics(df_counts, metric_cols=metric_cols, bucket_colname=bucket_colname)

    # Compute the final metrics
    select_exprs = [F.col(col) for col in df_counts.columns]
    if metric_type == "ate":
        target_rate_treated = F.col("treatment_target") / F.col("treatment_count")
        target_rate_control = F.col("control_target") / F.col("control_count")
        select_exprs += [
            target_rate_treated.alias("target_rate_treated"),
            target_rate_control.alias("target_rate_control"),
            (target_rate_treated - target_rate_control).alias("ate"),
        ]

    elif metric_type == "roi":
        select_exprs += [(F.col("tot_benefit") / F.col("tot_cost")).alias("roi")]

    elif metric_type == "iroi":
        scale_factor = F.col("treatment_count") / F.col("control_count")
        incremental_benefit = F.col("treatment_benefit") - F.col("control_benefit") * scale_factor
        incremental_cost = F.col("treatment_cost") - F.col("control_cost") * scale_factor
        select_exprs += [
            incremental_benefit.alias("incremental_benefit"),
            incremental_cost.alias("incremental_cost"),
            (incremental_benefit / incremental_cost).alias("iroi"),
        ]

    elif metric_type == "target_rate":
        select_exprs += [(F.col("tot_target") / F.col("count")).alias("target_rate")]

    df_counts = df_counts.select(*select_exprs)

    # Check if incremental cost is negative and flip the sign of the iROI in those cases
    if metric_type == "iroi":
        n_negative_cost_rows = df_counts.filter(F.col("incremental_cost") < 0).count()
        if n_negative_cost_rows > 0:
            print(
                f"Warning: {n_negative_cost_rows} rows had negative incremental cost. "
                f"iROI sign was flipped in those cases, but the results might be unreliable."
            )

    return df_counts


def _compute_ci(
    df: DataFrame,
    bucket_colname: Optional[str] = None,
    relevant_cols: Optional[List[str]] = None,
    ci_quantiles: Optional[List[float]] = None,
) -> DataFrame:
    """Computes confidence intervals from bootstrapped samples of the dataset. For more details on this procedure,
    see https://ocw.mit.edu/courses/mathematics/18-05-introduction-to-probability-and-statistics-spring-2014/
        readings/MIT18_05S14_Reading24.pdf.

    Args:
        df (pyspark.sql.DataFrame): a Spark dataframe
        bucket_colname (str, optional): the column name that contains the bucket assignments.
            If None, each row is assumed to represent the whole population.
        relevant_cols (list of str, optional): list of column names containing metrics
            to calculate confidence intervals for
        ci_quantiles (list of float, optional): the lower and upper confidence bounds.
            Only has an effect if bootstrap=True

    Returns:
        (pyspark.sql.DataFrame): a Spark dataframe containing point estimates and confidence intervals
            (aggregated over bootstraps)
    """

    if ci_quantiles is None:
        ci_quantiles = [0.025, 0.975]
    if relevant_cols is None:
        relevant_cols = ["ate"]

    group_cols = []
    select_cols: List[Column] = [F.col(col).alias(f"{col}_pe") for col in relevant_cols]
    if bucket_colname is not None:
        select_cols.append(F.col(bucket_colname))
        group_cols.append(bucket_colname)

    # fetch the point estimates
    point_estimates = df.filter(F.col("bootstrap") == 0).select(*select_cols)

    # calculate the diffs between bootstrapped estimates and the point estimates
    if bucket_colname is None:
        deltas = df.filter(F.col("bootstrap") != 0).crossJoin(point_estimates)
    else:
        deltas = df.filter(F.col("bootstrap") != 0).join(point_estimates, bucket_colname)
    for col in relevant_cols:
        deltas = deltas.withColumn(f"{col}_delta", F.col(col) - F.col(f"{col}_pe"))

    # get quantiles for the diffs
    agg_exprs = [
        F.expr(f"percentile_approx({col}_delta, {ci_quantiles[1]})").alias(f"{col}_delta_lower")
        for col in relevant_cols
    ]
    agg_exprs += [
        F.expr(f"percentile_approx({col}_delta, {ci_quantiles[0]})").alias(f"{col}_delta_upper")
        for col in relevant_cols
    ]
    agg_exprs += [F.first(F.col(f"{col}_pe")).alias(col) for col in relevant_cols]
    agg_exprs += [F.sqrt(F.mean(F.pow(F.col(f"{col}_delta"), 2))).alias(f"{col}_std_error") for col in relevant_cols]

    deltas_ci = deltas.groupby(group_cols).agg(*agg_exprs)

    # calculate upper and lower bounds of the estimates, based on the quantile values for the diffs
    for col in relevant_cols:
        deltas_ci = (
            deltas_ci.withColumn(f"{col}_lower", F.col(col) - F.col(f"{col}_delta_lower"))
            .withColumn(f"{col}_upper", F.col(col) - F.col(f"{col}_delta_upper"))
            .drop(f"{col}_delta_lower", f"{col}_delta_upper")
        )

    # add other relevant columns from the original sample
    if bucket_colname is None:
        deltas_ci = deltas_ci.crossJoin(df.filter(F.col("bootstrap") == 0).drop(*relevant_cols))
    else:
        deltas_ci = deltas_ci.join(df.filter(F.col("bootstrap") == 0).drop(*relevant_cols), bucket_colname)

    select_cols = [F.col(col) for col in df.columns if col != "bootstrap"]
    for relevant_col in relevant_cols:
        select_cols += [
            F.col(f"{relevant_col}_lower"),
            F.col(f"{relevant_col}_upper"),
            F.col(f"{relevant_col}_std_error"),
        ]

    return deltas_ci.select(select_cols)


def _add_fraction_of_population(df: DataFrame, bucket_colname: Optional[str] = None) -> DataFrame:
    """Adds the fraction of population column to df.

    Args:
        df (pyspark.sql.DataFrame): a Spark dataframe
        bucket_colname (str, optional): the column name that contains the bucket assignments.
            If None, each row is assumed to represent the whole population.

    Returns:
        (pyspark.sql.DataFrame): df with an additional column representing the fraction of population
    """
    bootstrap = "bootstrap" in df.columns

    # if there are no bucketing columns, set each fraction to 1
    if bucket_colname is None:
        df = df.withColumn("fraction", F.lit(1))

    # otherwise calculate the overall counts across all buckets and divide count (in each row) with this number
    else:
        _groupByCol: List[str] = ["bootstrap"] if bootstrap else []
        overall_counts = df.groupby(_groupByCol).agg(F.sum("count").alias("overall_count"))
        if bootstrap:
            df = df.join(overall_counts, "bootstrap")
        else:
            df = df.crossJoin(overall_counts)

        df = df.withColumn("fraction", F.col("count") / F.col("overall_count")).drop("overall_count")
    return df


def _add_cumulative_metrics(
    df: DataFrame, metric_cols: Optional[List[str]] = None, bucket_colname: Optional[str] = None
) -> DataFrame:
    """Adds columns with cumulative sums of each metric column in df.

    Args:
        df (pyspark.sql.DataFrame): a Spark dataframe containing metric values
        metric_cols (list of str, optional): column names for metrics that should be made cumulative
        bucket_colname (str, optional): the column name that contains the bucket assignments.
        The df will be ordered by this column descendingly before calculating the cumulative sums.

    Returns:
        (pyspark.sql.DataFrame): df with metrics changed to cumulative sums of the metrics
    """
    if bucket_colname is None:
        bucket_colname = "bucket"

    if metric_cols is None:
        metric_cols = []

    _partitionCol: List[str] = ["bootstrap"] if "bootstrap" in df.columns else []
    w = Window.partitionBy(*_partitionCol).orderBy(F.col(bucket_colname).desc())
    for col in metric_cols:
        df = df.withColumn(col, F.sum(col).over(w))

    return df


def _assign_bucket(
    df: DataFrame,
    n_buckets: int = 30,
    score_colname: str = "score",
    bucket_colname: str = "bucket",
    add_labels: bool = False,
) -> DataFrame:
    """Assigns instances to buckets according to model score quantiles.

    Args:
        df (pyspark.sql.DataFrame): a Spark dataframe containing real-valued model scores
        n_buckets (int, optional): the number of quantiles to generate from the column <score_colname>
        score_colname (str, optional): the column name in df that contains the model scores
        bucket_colname (str, optional): the column name that will contain bucket assignments
        add_labels (bool, optional): indicates whether the bucket labels are added in the form [start, end).
            Defaults to False, meaning that only the ids of the buckets are returned.

    Returns:
        (pyspark.sql.DataFrame): df with an added column with the bucket assignments
    """

    qds = QuantileDiscretizer(
        numBuckets=n_buckets,
        inputCol=score_colname,
        outputCol=bucket_colname,
        relativeError=0.00001,
        handleInvalid="skip",
    )

    qds_model = qds.fit(df)
    df_buckets = qds_model.transform(df)

    # add thresholds
    splits = qds_model.getSplits()
    bucket_label_mapping = {float(i): label for i, label in enumerate(splits[:-1])}  # type: ignore
    mapping_expr = F.create_map(*[F.lit(x) for x in itertools.chain(*bucket_label_mapping.items())])  # type: ignore
    df_buckets = df_buckets.withColumn("threshold", mapping_expr.getItem(F.col(bucket_colname)))

    if add_labels:
        # add the bucket label, showing the start and end points of the bucket
        bucket_labels = [f"[{start:.2f}, {end:.2f})" for start, end in zip(splits[:-1], splits[1:])]
        bucket_label_mapping = {float(i): label for i, label in enumerate(bucket_labels)}  # type: ignore
        mapping_expr = F.create_map(*[F.lit(x) for x in itertools.chain(*bucket_label_mapping.items())])  # type: ignore
        df_buckets = df_buckets.withColumn(bucket_colname, mapping_expr.getItem(F.col(bucket_colname)))

    return df_buckets


def estimate_ate(
    df: DataFrame,
    target_colname: str = "outcome",
    treatment_colname: str = "treatment",
    treatment_value: Union[str, int] = 1,
    control_value: Union[str, int] = 0,
    bootstrap: bool = False,
    n_bootstraps: int = 100,
    ci_quantiles: Optional[List[float]] = None,
) -> Dict:
    """Estimates the average treatment effect in a Spark DataFrame.

    Args:
        df (pyspark.sql.DataFrame): a dataframe containing the treatment indicators and observed outcomes
        target_colname (str, optional): the column name in df that contains the target
        treatment_colname (str, optional): the column name in df that contains the treatment indicators
        treatment_value (str or int, optional): the value in column <treatment_colname>
            that refers to the treatment group
        control_value (str or int, optional): the value in column <treatment_colname> that refers to the control group
        bootstrap (bool, optional): if True, will perform bootstrapping and return confidence intervals
        n_bootstraps (int, optional): the number of bootstraps to perform. Only has an effect if bootstrap=True
        ci_quantiles (list of float, optional): the lower and upper confidence bounds.
            Only has an effect if bootstrap=True

    Returns:
        Dict with estimates of the target rate in the control group, the target rate in the treatment group, and the ATE,
            all with or without lower and upper bounds depending on whether bootstrapping is performed.
    """
    if ci_quantiles is None:
        ci_quantiles = [0.025, 0.975]

    df_counts = _estimate_metrics(
        df,
        metric_type="ate",
        treatment_colname=treatment_colname,
        target_colname=target_colname,
        treatment_value=treatment_value,
        control_value=control_value,
        cumulative=False,
        bootstrap=bootstrap,
        n_bootstraps=n_bootstraps,
    )

    relevant_cols = ["target_rate_control", "target_rate_treated", "ate"]
    if bootstrap:
        # calculate confidence intervals
        df_counts = _compute_ci(df_counts, relevant_cols=relevant_cols, ci_quantiles=ci_quantiles)

        # ensure that the order of the returned values is right
        select_cols = []
        for relevant_col in relevant_cols:
            select_cols += [f"{relevant_col}_lower", relevant_col, f"{relevant_col}_upper"]

        return df_counts.select(select_cols).toPandas().to_dict(orient="records")[0]

    else:
        return df_counts.select(relevant_cols).toPandas().to_dict(orient="records")[0]


def estimate_roi(
    df: DataFrame,
    benefit_colname: str = "revenue",
    cost_colname: str = "cost",
    bootstrap: bool = False,
    n_bootstraps: int = 100,
    ci_quantiles: Optional[List[float]] = None,
) -> Dict:
    """Estimates the return on investment in a Spark DataFrame.

    Args:
        df (pyspark.sql.DataFrame): a dataframe containing the cost and the benefit for each instance
        benefit_colname (str, optional): the column name in df that contains the benefit
        cost_colname (str, optional): the column name in df that contains the cost
        bootstrap (bool, optional): if True, will perform bootstrapping and return confidence intervals
        n_bootstraps (int, optional): the number of bootstraps to perform. Only has an effect if bootstrap=True
        ci_quantiles (list of float, optional): the lower and upper confidence bounds.
            Only has an effect if bootstrap=True

    Returns:
        Dict with estimate of the ROI, with or without lower and upper bounds.
    """
    if ci_quantiles is None:
        ci_quantiles = [0.025, 0.975]
    df_counts = _estimate_metrics(
        df,
        metric_type="roi",
        benefit_colname=benefit_colname,
        cost_colname=cost_colname,
        cumulative=False,
        bootstrap=bootstrap,
        n_bootstraps=n_bootstraps,
    )

    relevant_cols = ["roi"]
    if bootstrap:
        # calculate confidence intervals
        df_counts = _compute_ci(df_counts, relevant_cols=relevant_cols, ci_quantiles=ci_quantiles)

        # ensure that the order of the returned values is right
        select_cols = []
        for relevant_col in relevant_cols:
            select_cols += [f"{relevant_col}_lower", relevant_col, f"{relevant_col}_upper"]

        return df_counts.select(select_cols).toPandas().to_dict(orient="records")[0]

    else:
        return df_counts.select(relevant_cols).toPandas().to_dict(orient="records")[0]


def estimate_iroi(
    df: DataFrame,
    benefit_colname: str = "revenue",
    cost_colname: str = "cost",
    treatment_colname: str = "treatment",
    treatment_value: Union[str, int] = 1,
    control_value: Union[str, int] = 0,
    bootstrap: bool = False,
    n_bootstraps: int = 100,
    ci_quantiles: Optional[List[float]] = None,
) -> Dict:
    """Estimates the incremental return on investment in a Spark DataFrame.

    Args:
        df (pyspark.sql.DataFrame): a dataframe containing the treatment indicators,
            the cost and the benefit for each instance
        benefit_colname (str, optional): the column name in df that contains the benefit
        cost_colname (str, optional): the column name in df that contains the cost
        treatment_colname (str, optional): the column name in df that contains the treatment indicators
        treatment_value (str or int, optional): the value in column <treatment_colname>
            that refers to the treatment group
        control_value (str or int, optional): the value in column <treatment_colname> that refers to the control group
        bootstrap (bool, optional): if True, will perform bootstrapping and return confidence intervals
        n_bootstraps (int, optional): the number of bootstraps to perform. Only has an effect if bootstrap=True
        ci_quantiles (list of float, optional): the lower and upper confidence bounds.
            Only has an effect if bootstrap=True

    Returns:
        Dict of estimates of the iROI, incremental benefit, and incremental cost, all with or without lower and upper bounds depending on whether bootstrapping is performed.
    """
    if ci_quantiles is None:
        ci_quantiles = [0.025, 0.975]

    df_counts = _estimate_metrics(
        df,
        metric_type="iroi",
        benefit_colname=benefit_colname,
        cost_colname=cost_colname,
        treatment_colname=treatment_colname,
        treatment_value=treatment_value,
        control_value=control_value,
        cumulative=False,
        bootstrap=bootstrap,
        n_bootstraps=n_bootstraps,
    )

    relevant_cols = ["iroi", "incremental_benefit", "incremental_cost"]
    if bootstrap:
        # calculate confidence intervals
        df_counts = _compute_ci(df_counts, relevant_cols=relevant_cols, ci_quantiles=ci_quantiles)

        # ensure that the order of the returned values is right
        select_cols = []
        for relevant_col in relevant_cols:
            select_cols += [f"{relevant_col}_lower", relevant_col, f"{relevant_col}_upper"]

        return df_counts.select(select_cols).toPandas().to_dict(orient="records")[0]

    else:
        return df_counts.select(relevant_cols).toPandas().to_dict(orient="records")[0]


def estimate_target_rate_per_bucket(
    df: DataFrame,
    bucket_colname: str = "bucket",
    target_colname: str = "outcome",
    bootstrap: bool = False,
    n_bootstraps: int = 100,
    ci_quantiles: Optional[List[float]] = None,
) -> pd.DataFrame:
    """Estimates conditional average treatment effects per bucket in a Spark DataFrame.

    Args:
        df (pyspark.sql.DataFrame): a dataframe containing the treatment indicators, the observed outcomes,
            and the bucket assignments
        bucket_colname (str, optional): column name in df that contains the bucket assignments
        target_colname (str, optional): the column name in df that contains the target
        bootstrap (bool, optional): if True, will perform bootstrapping and return confidence intervals
        n_bootstraps (int, optional): the number of bootstraps to perform. Only has an effect if bootstrap=True
        ci_quantiles (list of float, optional): the lower and upper confidence bounds.
            Only has an effect if bootstrap=True

    Returns:
        (pandas.DataFrame): a dataframe containing CATE estimates (with or without confidence intervals),
            population sizes and fractions within each bucket
    """

    if ci_quantiles is None:
        ci_quantiles = [0.025, 0.975]
    df_counts = _estimate_metrics(
        df,
        metric_type="target_rate",
        target_colname=target_colname,
        cumulative=False,
        bucket_colname=bucket_colname,
        bootstrap=bootstrap,
        n_bootstraps=n_bootstraps,
    )

    if bootstrap:
        df_counts = _compute_ci(
            df_counts, bucket_colname=bucket_colname, relevant_cols=["target_rate"], ci_quantiles=ci_quantiles
        )

    select_cols = [F.col(bucket_colname), F.col("count"), F.col("fraction")]
    select_cols += [F.col(col) for col in df_counts.columns if col.startswith("target_rate")]
    return df_counts.select(select_cols).orderBy(bucket_colname, ascending=False).toPandas()


def estimate_target_rate_per_quantile(
    df: DataFrame,
    n_buckets: int = 30,
    score_colname: str = "score",
    target_colname: str = "outcome",
    add_labels: bool = False,
    bootstrap: bool = False,
    n_bootstraps: int = 100,
    ci_quantiles: Optional[List[float]] = None,
) -> pd.DataFrame:
    """Divides the data into buckets based on model score quantiles and estimates average
        treatment effects per bucket.

    Args:
        df (pyspark.sql.DataFrame): a dataframe containing the treatment indicators,
            the observed outcomes, and real-valued model scores
        n_buckets (int, optional): the number of quantiles to generate from the column <score_colname>
        score_colname (str, optional): the column name in df that contains the model scores
        target_colname (str, optional): the column name in df that contains the target
        add_labels (bool, optional): indicates whether the bucket labels are added in the form [start, end).
            Defaults to False, meaning that only the ids of the buckets are returned.
        bootstrap (bool, optional): if True, will perform bootstrapping and return confidence intervals
        n_bootstraps (int, optional): the number of bootstraps to perform. Only has an effect if bootstrap=True
        ci_quantiles (list of float, optional): the lower and upper confidence bounds.
            Only has an effect if bootstrap=True

    Returns:
        (pandas.DataFrame): a dataframe containing CATE estimates (with or without confidence intervals),
            population sizes and fractions within each bucket
    """

    # Assign quantile bucket to each row
    if ci_quantiles is None:
        ci_quantiles = [0.025, 0.975]
    df_buckets = _assign_bucket(
        df, n_buckets=n_buckets, score_colname=score_colname, bucket_colname="bucket", add_labels=add_labels
    )

    return estimate_target_rate_per_bucket(
        df_buckets,
        bucket_colname="bucket",
        target_colname=target_colname,
        bootstrap=bootstrap,
        n_bootstraps=n_bootstraps,
        ci_quantiles=ci_quantiles,
    )


def plot_metric_per_bucket(
    df: pd.DataFrame, x: str = "bucket", y: str = "cate", bootstrap: bool = False, sort_x: bool = True, ax: Any = None
) -> Any:
    """Plots metric values per buckets as a barplot or scatterplot with errorbars.

    Args:
        df (pandas.DataFrame): a dataframe containing metric values per buckets. If bootstrap=True,
            df should also contain upper and lower bounds.
        x (str): column name in df that contains the bucket names. This defines the x-axis.
        y (str): column name in df that contains the metric values. This defines the y-axis.
        bootstrap (bool, optional): indicates whether to use lower and upper bound values from df and plot a
            scatterplot with errorbars. If False, plots a barplot.
        sort_x (bool, optional): if True, x-axis will be sorted from highest metric value to lowest
        ax (matplotlib.axes._subplots.AxesSubplot, optional): if specified, the plot will be plotted on this ax. Useful when creating a figure with subplots.

    Returns:
        (matplotlib.axes._subplots.AxesSubplot): the axis of the plot
    """

    df_metric = df.copy()

    # sort the buckets
    if sort_x:
        df_metric = df_metric.sort_values(y, ascending=False)

    if bootstrap:
        ax = sns.scatterplot(x=x, y=y, data=df_metric, ax=ax)

        # add error bars
        df_metric["lower"] = df_metric[y] - df_metric[f"{y}_lower"]
        df_metric["upper"] = df_metric[f"{y}_upper"] - df_metric[y]
        ax.errorbar(x=df_metric[x], y=df_metric[y], yerr=df_metric[["lower", "upper"]].values.T, fmt="ko")  # type: ignore

    else:
        ax = sns.barplot(x=x, y=y, data=df_metric, ax=ax)

    ax.set(title=f"{y} by bucket", xlabel=x, ylabel=y)

    return ax


def estimate_and_plot_target_rate_per_quantile(
    df: DataFrame,
    n_buckets: int = 30,
    score_colname: str = "cate_outcome",
    target_colname: str = "outcome",
    add_labels: bool = False,
    bootstrap: bool = False,
    n_bootstraps: int = 100,
    ci_quantiles: Optional[List[float]] = None,
    sort_x: bool = True,
    ax: Any = None,
) -> Tuple:
    """Divides the data into buckets based on model score quantiles, estimates CATE values per quantile,
        and plots them.

    Args:
        df (pyspark.sql.DataFrame): a dataframe containing the treatment indicators, the observed outcomes,
            and real-valued model scores
        n_buckets (int, optional): the number of quantiles to generate from the column <score_colname>
        score_colname (str, optional): the column name in df that contains the model scores
        target_colname (str, optional): the column name in df that contains the target
        add_labels (bool, optional): indicates whether the bucket labels are added in the form [start, end).
            Defaults to False, meaning that only the ids of the buckets are returned.
        bootstrap (bool, optional): if True, will perform bootstrapping and return confidence intervals
        n_bootstraps (int, optional): the number of bootstraps to perform. Only has an effect if bootstrap=True
        ci_quantiles (list of float, optional): the lower and upper confidence bounds.
            Only has an effect if bootstrap=True
        sort_x (bool, optional): if True, x-axis will be sorted from highest metric value to lowest
        ax (matplotlib.axes._subplots.AxesSubplot, optional): if specified, the plot will be plotted on this ax. Useful when creating a figure with subplots.

    Returns:
        A tuple, containing:
        (pandas.DataFrame): a dataframe containing the target rate estimates (with or without confidence intervals),
            population sizes and fractions within each bucket
        (matplotlib.axes._subplots.AxesSubplot): the axis of the plot
    """

    if ci_quantiles is None:
        ci_quantiles = [0.025, 0.975]
    quantile_df = estimate_target_rate_per_quantile(
        df=df,
        n_buckets=n_buckets,
        target_colname=target_colname,
        score_colname=score_colname,
        add_labels=add_labels,
        bootstrap=bootstrap,
        n_bootstraps=n_bootstraps,
        ci_quantiles=ci_quantiles,
    )

    ax = plot_metric_per_bucket(quantile_df, y="target_rate", bootstrap=bootstrap, sort_x=sort_x, ax=ax)

    return quantile_df, ax


def estimate_and_plot_target_rate_per_bucket(
    df: DataFrame,
    bucket_colname: str = "bucket",
    target_colname: str = "outcome",
    bootstrap: bool = False,
    n_bootstraps: int = 100,
    ci_quantiles: Optional[List[float]] = None,
    sort_x: bool = True,
    ax: Any = None,
) -> Tuple:
    """Estimates conditional average treatment effects per bucket in a Spark DataFrame and plots the estimates.

    Args:
        df (pyspark.sql.DataFrame): a dataframe containing the treatment indicators, the observed outcomes,
            and real-valued model scores
        bucket_colname (str, optional): column name in df that contains the bucket assignments
        target_colname (str, optional): the column name in df that contains the target
            Defaults to False, meaning that only the ids of the buckets are returned.
        bootstrap (bool, optional): if True, will perform bootstrapping and return confidence intervals
        n_bootstraps (int, optional): the number of bootstraps to perform. Only has an effect if bootstrap=True
        ci_quantiles (list of float, optional): the lower and upper confidence bounds.
            Only has an effect if bootstrap=True
        sort_x (bool, optional): if True, x-axis will be sorted from highest metric value to lowest
        ax (matplotlib.axes._subplots.AxesSubplot, optional): if specified, the plot will be plotted on this ax. Useful when creating a figure with subplots.

    Returns:
        A tuple, containing:
        (pandas.DataFrame): a dataframe containing CATE estimates (with or without confidence intervals),
            population sizes and fractions within each bucket
        (matplotlib.axes._subplots.AxesSubplot): the axis of the plot
    """

    if ci_quantiles is None:
        ci_quantiles = [0.025, 0.975]
    quantile_df = estimate_target_rate_per_bucket(
        df=df,
        bucket_colname=bucket_colname,
        target_colname=target_colname,
        bootstrap=bootstrap,
        n_bootstraps=n_bootstraps,
        ci_quantiles=ci_quantiles,
    )

    ax = plot_metric_per_bucket(quantile_df, y="target_rate", bootstrap=bootstrap, sort_x=sort_x, ax=ax)

    return quantile_df, ax


def estimate_cate_per_bucket(
    df: DataFrame,
    bucket_colname: str = "bucket",
    target_colname: str = "outcome",
    treatment_colname: str = "treatment",
    treatment_value: Union[str, int] = 1,
    control_value: Union[str, int] = 0,
    bootstrap: bool = False,
    n_bootstraps: int = 100,
    ci_quantiles: Optional[List[float]] = None,
) -> pd.DataFrame:
    """Estimates the conditional average treatment effects per bucket in a Spark DataFrame.

    Args:
        df (pyspark.sql.DataFrame): a dataframe containing the treatment indicators, the observed outcomes,
        and the bucket assignments
        bucket_colname (str, optional): column name in df that contains the bucket assignments
        target_colname (str, optional): the column name in df that contains the target
        treatment_colname (str, optional): the column name in df that contains the treatment indicators
        treatment_value (str or int, optional): the value in column <treatment_colname> that refers to the
            treatment group
        control_value (str or int, optional): the value in column <treatment_colname> that refers to the control group
        bootstrap (bool, optional): if True, will perform bootstrapping and return confidence intervals
        n_bootstraps (int, optional): the number of bootstraps to perform. Only has an effect if bootstrap=True
        ci_quantiles (list of float, optional): the lower and upper confidence bounds.
            Only has an effect if bootstrap=True

    Returns:
        (pandas.DataFrame): a dataframe containing CATE estimates (with or without confidence intervals),
            population sizes and fractions within each bucket
    """

    if ci_quantiles is None:
        ci_quantiles = [0.025, 0.975]
    df_counts = _estimate_metrics(
        df,
        metric_type="ate",
        treatment_colname=treatment_colname,
        target_colname=target_colname,
        treatment_value=treatment_value,
        control_value=control_value,
        cumulative=False,
        bucket_colname=bucket_colname,
        bootstrap=bootstrap,
        n_bootstraps=n_bootstraps,
    )

    if bootstrap:
        df_counts = _compute_ci(
            df_counts, bucket_colname=bucket_colname, relevant_cols=["ate"], ci_quantiles=ci_quantiles
        )

    select_cols = [F.col(bucket_colname), F.col("count"), F.col("fraction")]
    select_cols += [F.col(col).alias(col.replace("ate", "cate")) for col in df_counts.columns if col.startswith("ate")]
    return df_counts.select(select_cols).orderBy(bucket_colname, ascending=False).toPandas()


def estimate_cate_per_quantile(
    df: DataFrame,
    n_buckets: int = 30,
    score_colname: str = "cate_outcome",
    target_colname: str = "outcome",
    treatment_colname: str = "treatment",
    treatment_value: Union[str, int] = 1,
    control_value: Union[str, int] = 0,
    add_labels: bool = False,
    bootstrap: bool = False,
    n_bootstraps: int = 100,
    ci_quantiles: Optional[List[float]] = None,
) -> pd.DataFrame:
    """Divides the data into buckets based on model score quantiles and estimates average treatment
        effects per bucket.

    Args:
        df (pyspark.sql.DataFrame): a dataframe containing the treatment indicators,
            the observed outcomes, and real-valued model scores
        n_buckets (int, optional): the number of quantiles to generate from the column <score_colname>
        score_colname (str, optional): the column name in df that contains the model scores
        target_colname (str, optional): the column name in df that contains the target
        treatment_colname (str, optional): the column name in df that contains the treatment indicators
        treatment_value (str or int, optional): the value in column <treatment_colname>
            that refers to the treatment group
        control_value (str or int, optional): the value in column <treatment_colname> that refers to the control group
        add_labels (bool, optional): indicates whether the bucket labels are added in the form [start, end).
            Defaults to False, meaning that only the ids of the buckets are returned.
        bootstrap (bool, optional): if True, will perform bootstrapping and return confidence intervals
        n_bootstraps (int, optional): the number of bootstraps to perform. Only has an effect if bootstrap=True
        ci_quantiles (list of float, optional): the lower and upper confidence bounds.
            Only has an effect if bootstrap=True

    Returns:
        (pandas.DataFrame): a dataframe containing CATE estimates (with or without confidence intervals),
            population sizes and fractions within each bucket
    """

    # Assign quantile bucket to each row
    if ci_quantiles is None:
        ci_quantiles = [0.025, 0.975]
    df_buckets = _assign_bucket(
        df, n_buckets=n_buckets, score_colname=score_colname, bucket_colname="bucket", add_labels=add_labels
    )

    return estimate_cate_per_bucket(
        df_buckets,
        bucket_colname="bucket",
        treatment_colname=treatment_colname,
        target_colname=target_colname,
        treatment_value=treatment_value,
        control_value=control_value,
        bootstrap=bootstrap,
        n_bootstraps=n_bootstraps,
        ci_quantiles=ci_quantiles,
    )


def estimate_and_plot_cate_per_quantile(
    df: DataFrame,
    n_buckets: int = 30,
    score_colname: str = "cate_outcome",
    target_colname: str = "outcome",
    treatment_colname: str = "treatment",
    treatment_value: Union[str, int] = 1,
    control_value: Union[str, int] = 0,
    add_labels: bool = False,
    bootstrap: bool = False,
    n_bootstraps: int = 100,
    ci_quantiles: Optional[List[float]] = None,
    sort_x: bool = True,
    ax: Any = None,
) -> Tuple:
    """Divides the data into buckets based on model score quantiles, estimates CATE values per quantile,
        and plots them.

    Args:
        df (pyspark.sql.DataFrame): a dataframe containing the treatment indicators, the observed outcomes,
            and real-valued model scores
        n_buckets (int, optional): the number of quantiles to generate from the column <score_colname>
        score_colname (str, optional): the column name in df that contains the model scores
        target_colname (str, optional): the column name in df that contains the target
        treatment_colname (str, optional): the column name in df that contains the treatment indicators
        treatment_value (str or int, optional): the value in column <treatment_colname>
            that refers to the treatment group
        control_value (str or int, optional): the value in column <treatment_colname> that refers to the control group
        add_labels (bool, optional): indicates whether the bucket labels are added in the form [start, end).
            Defaults to False, meaning that only the ids of the buckets are returned.
        bootstrap (bool, optional): if True, will perform bootstrapping and return confidence intervals
        n_bootstraps (int, optional): the number of bootstraps to perform. Only has an effect if bootstrap=True
        ci_quantiles (list of float, optional): the lower and upper confidence bounds.
            Only has an effect if bootstrap=True
        sort_x (bool, optional): if True, x-axis will be sorted from highest metric value to lowest
        ax (matplotlib.axes._subplots.AxesSubplot, optional): if specified, the plot will be plotted on this ax. Useful when creating a figure with subplots.

    Returns:
        A tuple, containing:
        (pandas.DataFrame): a dataframe containing CATE estimates (with or without confidence intervals),
            population sizes and fractions within each bucket
        (matplotlib.axes._subplots.AxesSubplot): the axis of the plot
    """

    if ci_quantiles is None:
        ci_quantiles = [0.025, 0.975]
    quantile_df = estimate_cate_per_quantile(
        df=df,
        n_buckets=n_buckets,
        treatment_colname=treatment_colname,
        target_colname=target_colname,
        treatment_value=treatment_value,
        control_value=control_value,
        score_colname=score_colname,
        add_labels=add_labels,
        bootstrap=bootstrap,
        n_bootstraps=n_bootstraps,
        ci_quantiles=ci_quantiles,
    )

    ax = plot_metric_per_bucket(quantile_df, bootstrap=bootstrap, sort_x=sort_x, ax=ax)

    return quantile_df, ax


def estimate_and_plot_cate_per_bucket(
    df: DataFrame,
    bucket_colname: str = "bucket",
    target_colname: str = "outcome",
    treatment_colname: str = "treatment",
    treatment_value: Union[str, int] = 1,
    control_value: Union[str, int] = 0,
    bootstrap: bool = False,
    n_bootstraps: int = 100,
    ci_quantiles: Optional[List[float]] = None,
    sort_x: bool = True,
    ax: Any = None,
) -> Tuple:
    """Estimates conditional average treatment effects per bucket in a Spark DataFrame and plots the estimates.

    Args:
        df (pyspark.sql.DataFrame): a dataframe containing the treatment indicators, the observed outcomes,
            and real-valued model scores
        bucket_colname (str, optional): column name in df that contains the bucket assignments
        target_colname (str, optional): the column name in df that contains the target
        treatment_colname (str, optional): the column name in df that contains the treatment indicators
        treatment_value (str or int, optional): the value in column <treatment_colname>
            that refers to the treatment group
        control_value (str or int, optional): the value in column <treatment_colname> that refers to the control group
            Defaults to False, meaning that only the ids of the buckets are returned.
        bootstrap (bool, optional): if True, will perform bootstrapping and return confidence intervals
        n_bootstraps (int, optional): the number of bootstraps to perform. Only has an effect if bootstrap=True
        ci_quantiles (list of float, optional): the lower and upper confidence bounds.
            Only has an effect if bootstrap=True
        sort_x (bool, optional): if True, x-axis will be sorted from highest metric value to lowest
        ax (matplotlib.axes._subplots.AxesSubplot, optional): if specified, the plot will be plotted on this ax. Useful when creating a figure with subplots.

    Returns:
        A tuple, containing:
        (pandas.DataFrame): a dataframe containing CATE estimates (with or without confidence intervals),
            population sizes and fractions within each bucket
        (matplotlib.axes._subplots.AxesSubplot): the axis of the plot
    """

    if ci_quantiles is None:
        ci_quantiles = [0.025, 0.975]
    quantile_df = estimate_cate_per_bucket(
        df=df,
        bucket_colname=bucket_colname,
        treatment_colname=treatment_colname,
        target_colname=target_colname,
        treatment_value=treatment_value,
        control_value=control_value,
        bootstrap=bootstrap,
        n_bootstraps=n_bootstraps,
        ci_quantiles=ci_quantiles,
    )

    ax = plot_metric_per_bucket(quantile_df, bootstrap=bootstrap, sort_x=sort_x, ax=ax)

    return quantile_df, ax


def estimate_iroi_per_bucket(
    df: DataFrame,
    bucket_colname: str = "bucket",
    benefit_colname: str = "revenue",
    cost_colname: str = "cost",
    treatment_colname: str = "treatment",
    treatment_value: Union[int, str] = 1,
    control_value: Union[int, str] = 0,
    bootstrap: bool = False,
    n_bootstraps: int = 100,
    ci_quantiles: Optional[List[float]] = None,
) -> pd.DataFrame:
    """Estimates incremental ROI per bucket in a Spark DataFrame.

    Args:
        df (pyspark.sql.DataFrame): a dataframe containing the treatment indicators, benefit and cost,
            and the bucket assignments
        bucket_colname (str, optional): column name in df that contains the bucket assignments
        benefit_colname (str, optional): the column name in df that contains the benefit
        cost_colname (str, optional): the column name in df that contains the cost
        treatment_colname (str, optional): the column name in df that contains the treatment indicators
        treatment_value (str or int, optional): the value in column <treatment_colname> that refers to the
            treatment group
        control_value (str or int, optional): the value in column <treatment_colname> that refers to the control group
        bootstrap (bool, optional): if True, will perform bootstrapping and return confidence intervals
        n_bootstraps (int, optional): the number of bootstraps to perform. Only has an effect if bootstrap=True
        ci_quantiles (list of float, optional): the lower and upper confidence bounds.
            Only has an effect if bootstrap=True

    Returns:
        (pandas.DataFrame): a dataframe containing iROI estimates (with or without confidence intervals),
            population sizes and fractions within each bucket
    """

    if ci_quantiles is None:
        ci_quantiles = [0.025, 0.975]
    df_counts = _estimate_metrics(
        df,
        metric_type="iroi",
        benefit_colname=benefit_colname,
        cost_colname=cost_colname,
        treatment_colname=treatment_colname,
        treatment_value=treatment_value,
        control_value=control_value,
        cumulative=False,
        bucket_colname=bucket_colname,
        bootstrap=bootstrap,
        n_bootstraps=n_bootstraps,
    )

    if bootstrap:
        df_counts = _compute_ci(
            df_counts, bucket_colname=bucket_colname, relevant_cols=["iroi"], ci_quantiles=ci_quantiles
        )

    select_cols = [F.col(bucket_colname), F.col("count"), F.col("fraction")]
    select_cols += [F.col(col) for col in df_counts.columns if col.startswith("iroi")]
    return df_counts.select(select_cols).orderBy(bucket_colname, ascending=False).toPandas()


def estimate_iroi_per_quantile(
    df: DataFrame,
    n_buckets: int = 30,
    score_colname: str = "score",
    benefit_colname: str = "revenue",
    cost_colname: str = "cost",
    treatment_colname: str = "treatment",
    treatment_value: Union[str, int] = 1,
    control_value: Union[str, int] = 0,
    add_labels: bool = False,
    bootstrap: bool = False,
    n_bootstraps: int = 100,
    ci_quantiles: Optional[List[float]] = None,
) -> pd.DataFrame:
    """Divides the data into buckets based on model score quantiles and estimates iROI per bucket.

    Args:
        df (pyspark.sql.DataFrame): a dataframe containing the treatment indicators, benefit and cost,
            and real-valued model scores
        n_buckets (int, optional): the number of quantiles to generate from the column <score_colname>
        score_colname (str, optional): the column name in df that contains the model scores
        benefit_colname (str, optional): the column name in df that contains the benefit
        cost_colname (str, optional): the column name in df that contains the cost
        treatment_colname (str, optional): the column name in df that contains the treatment indicators
        treatment_value (str or int, optional): the value in column <treatment_colname> that
            refers to the treatment group
        control_value (str or int, optional): the value in column <treatment_colname> that refers to the control group
        add_labels (bool, optional): indicates whether the bucket labels are added in the form [start, end).
            Defaults to False, meaning that only the ids of the buckets are returned.
        bootstrap (bool, optional): if True, will perform bootstrapping and return confidence intervals
        n_bootstraps (int, optional): the number of bootstraps to perform. Only has an effect if bootstrap=True
        ci_quantiles (list of float, optional): the lower and upper confidence bounds.
            Only has an effect if bootstrap=True

    Returns:
        (pandas.DataFrame): a dataframe containing iROI estimates (with or without confidence intervals),
            population sizes and fractions within each bucket
    """

    # Assign quantile bucket to each row
    if ci_quantiles is None:
        ci_quantiles = [0.025, 0.975]
    df_buckets = _assign_bucket(
        df, n_buckets=n_buckets, score_colname=score_colname, bucket_colname="bucket", add_labels=add_labels
    )

    return estimate_iroi_per_bucket(
        df_buckets,
        bucket_colname="bucket",
        treatment_colname=treatment_colname,
        benefit_colname=benefit_colname,
        cost_colname=cost_colname,
        treatment_value=treatment_value,
        control_value=control_value,
        bootstrap=bootstrap,
        n_bootstraps=n_bootstraps,
        ci_quantiles=ci_quantiles,
    )


def estimate_and_plot_iroi_per_bucket(
    df: DataFrame,
    bucket_colname: str = "bucket",
    benefit_colname: str = "revenue",
    cost_colname: str = "cost",
    treatment_colname: str = "treatment",
    treatment_value: Union[str, int] = 1,
    control_value: Union[str, int] = 0,
    bootstrap: bool = False,
    n_bootstraps: int = 100,
    ci_quantiles: Optional[List[float]] = None,
    sort_x: bool = True,
    ax: Any = None,
) -> Tuple:
    """Estimates incremental ROI per bucket in a Spark DataFrame and plots the estimates.

    Args:
        df (pyspark.sql.DataFrame): a dataframe containing the treatment indicators, benefit and cost,
            and real-valued model scores
        bucket_colname (str, optional): column name in df that contains the bucket assignments
        benefit_colname (str, optional): the column name in df that contains the benefit
        cost_colname (str, optional): the column name in df that contains the cost
        treatment_colname (str, optional): the column name in df that contains the treatment indicators
        treatment_value (str or int, optional): the value in column <treatment_colname>
            that refers to the treatment group
        control_value (str or int, optional): the value in column <treatment_colname> that refers to the control group
        bootstrap (bool, optional): if True, will perform bootstrapping and return confidence intervals
        n_bootstraps (int, optional): the number of bootstraps to perform. Only has an effect if bootstrap=True
        ci_quantiles (list of float, optional): the lower and upper confidence bounds.
            Only has an effect if bootstrap=True
        sort_x (bool, optional): if True, x-axis will be sorted from highest metric value to lowest
        ax (matplotlib.axes._subplots.AxesSubplot, optional): if specified, the plot will be plotted on this ax. Useful when creating a figure with subplots.

    Returns:
        A tuple, containing:
        (pandas.DataFrame): a dataframe containing iROI estimates (with or without confidence intervals),
            population sizes and fractions within each bucket
        (matplotlib.axes._subplots.AxesSubplot): the axis of the plot
    """

    if ci_quantiles is None:
        ci_quantiles = [0.025, 0.975]
    quantile_df = estimate_iroi_per_bucket(
        df,
        bucket_colname=bucket_colname,
        benefit_colname=benefit_colname,
        cost_colname=cost_colname,
        treatment_colname=treatment_colname,
        treatment_value=treatment_value,
        control_value=control_value,
        bootstrap=bootstrap,
        n_bootstraps=n_bootstraps,
        ci_quantiles=ci_quantiles,
    )

    ax = plot_metric_per_bucket(quantile_df, x="bucket", y="iroi", bootstrap=bootstrap, sort_x=sort_x, ax=ax)

    return df, ax


def estimate_and_plot_iroi_per_quantile(
    df: DataFrame,
    n_buckets: int = 30,
    score_colname: str = "score",
    benefit_colname: str = "revenue",
    cost_colname: str = "cost",
    treatment_colname: str = "treatment",
    treatment_value: Union[str, int] = 1,
    control_value: Union[str, int] = 0,
    add_labels: bool = False,
    bootstrap: bool = False,
    n_bootstraps: int = 100,
    ci_quantiles: Optional[List[float]] = None,
    sort_x: bool = True,
    ax: Any = None,
) -> Tuple:
    """Divides the data into buckets based on model score quantiles, estimates iROI per quantile,
        and plots the estimates.

    Args:
        df (pyspark.sql.DataFrame): a dataframe containing the treatment indicators, benefit and cost,
            and real-valued model scores
        n_buckets (int, optional): the number of quantiles to generate from the column <score_colname>
        score_colname (str, optional): the column name in df that contains the model scores
        benefit_colname (str, optional): the column name in df that contains the benefit
        cost_colname (str, optional): the column name in df that contains the cost
        treatment_colname (str, optional): the column name in df that contains the treatment indicators
        treatment_value (str or int, optional): the value in column <treatment_colname>
            that refers to the treatment group
        control_value (str or int, optional): the value in column <treatment_colname> that refers to the control group
        add_labels (bool, optional): indicates whether the bucket labels are added in the form [start, end).
            Defaults to False, meaning that only the ids of the buckets are returned.
        bootstrap (bool, optional): if True, will perform bootstrapping and return confidence intervals
        n_bootstraps (int, optional): the number of bootstraps to perform. Only has an effect if bootstrap=True
        ci_quantiles (list of float, optional): the lower and upper confidence bounds.
            Only has an effect if bootstrap=True
        sort_x (bool, optional): if True, x-axis will be sorted from highest metric value to lowest
        ax (matplotlib.axes._subplots.AxesSubplot, optional): if specified, the plot will be plotted on this ax. Useful when creating a figure with subplots.

    Returns:
        A tuple, containing:
        (pandas.DataFrame): a dataframe containing iROI estimates (with or without confidence intervals),
            population sizes and fractions within each bucket
        (matplotlib.axes._subplots.AxesSubplot): the axis of the plot
    """

    if ci_quantiles is None:
        ci_quantiles = [0.025, 0.975]
    quantile_df = estimate_iroi_per_quantile(
        df,
        n_buckets=n_buckets,
        benefit_colname=benefit_colname,
        cost_colname=cost_colname,
        treatment_colname=treatment_colname,
        treatment_value=treatment_value,
        control_value=control_value,
        score_colname=score_colname,
        bootstrap=bootstrap,
        n_bootstraps=n_bootstraps,
        ci_quantiles=ci_quantiles,
        add_labels=add_labels,
    )

    ax = plot_metric_per_bucket(quantile_df, x="bucket", y="iroi", bootstrap=bootstrap, sort_x=sort_x, ax=ax)

    return quantile_df, ax


def estimate_cate_lift(
    df: DataFrame,
    n_buckets: int = 30,
    score_colname: str = "cate_outcome",
    target_colname: str = "outcome",
    treatment_colname: str = "treatment",
    treatment_value: Union[str, int] = 1,
    control_value: Union[str, int] = 0,
    bootstrap: bool = False,
    n_bootstraps: int = 100,
    ci_quantiles: Optional[List[float]] = None,
) -> pd.DataFrame:
    """Divides the data into buckets based on model score quantiles and cumulatively estimates CATE lift
        (with or without confidence intervals).

    Args:
        df (pyspark.sql.DataFrame): a dataframe containing the treatment indicators, the observed outcomes,
            and real-valued model scores
        n_buckets (int, optional): the number of quantiles to generate from the column <score_colname>
        score_colname (str, optional): the column name in df that contains the model scores
        target_colname (str, optional): the column name in df that contains the target
        treatment_colname (str, optional): the column name in df that contains the treatment indicators
        treatment_value (str or int, optional): the value in column <treatment_colname> that refers to the
            treatment group
        control_value (str or int, optional): the value in column <treatment_colname> that refers to the control group
        bootstrap (bool, optional): if True, will perform bootstrapping and return confidence intervals
        n_bootstraps (int, optional): the number of bootstraps to perform. Only has an effect if bootstrap=True
        ci_quantiles (list of float, optional): the lower and upper confidence bounds.
            Only has an effect if bootstrap=True

    Returns:
        (pandas.DataFrame): a dataframe containing the CATE lift estimates (with or without confidence intervals),
            cumulative population sizes and fractions
    """

    # Assign quantile bucket to each row
    if ci_quantiles is None:
        ci_quantiles = [0.025, 0.975]
    bucket_colname = "bucket"
    df_buckets = _assign_bucket(
        df, n_buckets=n_buckets, score_colname=score_colname, bucket_colname=bucket_colname, add_labels=False
    )

    # Estimate metrics, possibly bootstrap
    df_counts = _estimate_metrics(
        df_buckets,
        metric_type="ate",
        treatment_colname=treatment_colname,
        target_colname=target_colname,
        treatment_value=treatment_value,
        control_value=control_value,
        cumulative=True,
        bucket_colname=bucket_colname,
        bootstrap=bootstrap,
        n_bootstraps=n_bootstraps,
    )

    if bootstrap:
        df_counts = _compute_ci(
            df_counts, bucket_colname=bucket_colname, relevant_cols=["ate"], ci_quantiles=ci_quantiles
        )

    select_cols = [F.col(bucket_colname), F.col("count"), F.col("fraction")]
    select_cols += [
        F.col(col).alias(col.replace("ate", "cum_cate")) for col in df_counts.columns if col.startswith("ate")
    ]
    return df_counts.select(select_cols).orderBy(bucket_colname, ascending=False).toPandas()


def plot_cate_lift(
    df: pd.DataFrame,
    x: str = "fraction",
    y: str = "cum_cate",
    label: Optional[str] = None,
    bootstrap: bool = False,
    ax: Any = None,
) -> Any:
    """Plots the CATE lift estimates as a lineplot.

    Args:
        df (pandas.DataFrame): a dataframe containing CATE lift estimates and cumulative population sizes or fractions.
            If bootstrap=True, df should also contain upper and lower bounds for the lift estimates.
        x (str, optional): column name in df that contains the cumulative population sizes or fractions.
            This defines the x-axis.
        y (str, optional): column name in df that contains the CATE lift estimates. This defines the y-axis.
        label (str, optional): name of the score estimation method to be shown on the legend. Defaults to None.
        bootstrap (bool, optional): indicates whether to use lower and upper bound values from df and plot a
            scatterplot with errorbars. If False, plots a barplot.
        ax (matplotlib.axes._subplots.AxesSubplot, optional): if specified, the plot will be plotted on this ax. Useful when creating a figure with subplots.

    Returns:
        (matplotlib.axes._subplots.AxesSubplot): the axis of the plot
    """

    df_lift = df.copy()

    ax = sns.lineplot(x=x, y=y, data=df_lift, label=label, ax=ax)

    if bootstrap:
        # add error bars
        df_lift["lower"] = df_lift[y] - df_lift[f"{y}_lower"]
        df_lift["upper"] = df_lift[f"{y}_upper"] - df_lift[y]
        ax.errorbar(x=df_lift[x], y=df_lift[y], yerr=df_lift[["lower", "upper"]].values.T, fmt="ko")

    xlabels = [f"{x:,.0f}" for x in ax.get_xticks() * 100]
    _ = ax.set_xticklabels(xlabels)
    ax.set(title="CATE lift by quantile", xlabel="% population targeted", ylabel="ATE over the targeted population")
    return ax


def estimate_and_plot_cate_lift(
    df: DataFrame,
    n_buckets: int = 30,
    score_colname: str = "cate_outcome",
    treatment_colname: str = "treatment",
    target_colname: str = "outcome",
    treatment_value: Union[str, int] = 1,
    control_value: Union[str, int] = 0,
    label: Optional[str] = None,
    bootstrap: bool = False,
    n_bootstraps: int = 100,
    ci_quantiles: Optional[List[float]] = None,
    ax: Any = None,
) -> Tuple:
    """Divides the data into buckets based on model score quantiles, cumulatively estimates CATE lift
        (with or without confidence intervals), and plots the estimates as a lineplot.

    Args:
        df (pyspark.sql.DataFrame): a dataframe containing the treatment indicators, the observed outcomes,
            and real-valued model scores
        n_buckets (int, optional): the number of quantiles to generate from the column <score_colname>
        score_colname (str, optional): the column name in df that contains the model scores
        treatment_colname (str, optional): the column name in df that contains the treatment indicators
        target_colname (str, optional): the column name in df that contains the target
        treatment_value (str or int, optional): the value in column <treatment_colname>
            that refers to the treatment group
        control_value (str or int, optional): the value in column <treatment_colname> that refers to the control group
        label (str, optional): name of the score estimation method to be shown on the legend. Defaults to None.
        bootstrap (bool, optional): if True, will perform bootstrapping and return confidence intervals
        n_bootstraps (int, optional): the number of bootstraps to perform. Only has an effect if bootstrap=True
        ci_quantiles (list of float, optional): the lower and upper confidence bounds.
            Only has an effect if bootstrap=True
        ax (matplotlib.axes._subplots.AxesSubplot, optional): if specified, the plot will be plotted on this ax. Useful when creating a figure with subplots.

    Returns:
        A tuple, containing:
        (pandas.DataFrame): a dataframe containing the CATE lift estimates (with or without confidence intervals),
            cumulative population sizes and fractions
        (matplotlib.axes._subplots.AxesSubplot): the axis of the plot
    """

    if ci_quantiles is None:
        ci_quantiles = [0.025, 0.975]
    df_lift = estimate_cate_lift(
        df=df,
        n_buckets=n_buckets,
        treatment_colname=treatment_colname,
        target_colname=target_colname,
        treatment_value=treatment_value,
        control_value=control_value,
        score_colname=score_colname,
        bootstrap=bootstrap,
        n_bootstraps=n_bootstraps,
        ci_quantiles=ci_quantiles,
    )

    ax = plot_cate_lift(df_lift, label=label, bootstrap=bootstrap, ax=ax)

    return df_lift, ax


def estimate_qini(
    df: DataFrame,
    n_buckets: int = 30,
    score_colname: str = "cate_outcome",
    treatment_colname: str = "treatment",
    target_colname: str = "outcome",
    treatment_value: Union[str, int] = 1,
    control_value: Union[str, int] = 0,
    bootstrap: bool = False,
    n_bootstraps: int = 100,
    ci_quantiles: Optional[List[float]] = None,
) -> pd.DataFrame:
    """Divides the data into buckets based on model score quantiles and estimates Qini values
        (with or without confidence intervals).

    Args:
        df (pyspark.sql.DataFrame): a dataframe containing the treatment indicators, the observed outcomes,
            and real-valued model scores
        n_buckets (int, optional): the number of quantiles to generate from the column <score_colname>
        score_colname (str, optional): the column name in df that contains the model scores
        treatment_colname (str, optional): the column name in df that contains the treatment indicators
        target_colname (str, optional): the column name in df that contains the target
        treatment_value (str or int, optional): the value in column <treatment_colname>
            hat refers to the treatment group
        control_value (str or int, optional): the value in column <treatment_colname> that refers to the control group
        bootstrap (bool, optional): if True, will perform bootstrapping and return confidence intervals
        n_bootstraps (int, optional): the number of bootstraps to perform. Only has an effect if bootstrap=True
        ci_quantiles (list of float, optional): the lower and upper confidence bounds.
            Only has an effect if bootstrap=True

    Returns:
        (pandas.DataFrame): a dataframe containing Qini estimates (with or without confidence intervals),
            cumulative population sizes and fractions
    """

    # Assign quantile bucket to each row
    if ci_quantiles is None:
        ci_quantiles = [0.025, 0.975]
    bucket_colname = "bucket"
    df_buckets = _assign_bucket(
        df, n_buckets=n_buckets, score_colname=score_colname, bucket_colname=bucket_colname, add_labels=False
    )

    # Estimate metrics, possibly bootstrap
    df_counts = _estimate_metrics(
        df_buckets,
        metric_type="ate",
        treatment_colname=treatment_colname,
        target_colname=target_colname,
        treatment_value=treatment_value,
        control_value=control_value,
        cumulative=True,
        bucket_colname=bucket_colname,
        bootstrap=bootstrap,
        n_bootstraps=n_bootstraps,
    )
    # replace cate metric with qini
    df_counts = df_counts.withColumn("ate", F.col("ate") * F.col("fraction"))

    if bootstrap:
        df_counts = _compute_ci(
            df_counts, bucket_colname=bucket_colname, relevant_cols=["ate"], ci_quantiles=ci_quantiles
        )

    select_cols = [bucket_colname, "count", "fraction", "threshold"]
    select_cols += [col for col in df_counts.columns if col.startswith("ate")]
    df_pandas = df_counts.select(select_cols).orderBy(bucket_colname, ascending=False).toPandas()

    # add a zero-row, for the first point in a Qini curve
    zero_row = {col: 0 for col in df_pandas.columns}
    zero_row["threshold"] = np.inf  # type: ignore
    zero_row["bucket"] = None  # type: ignore
    zero_row = pd.DataFrame(zero_row, index=[0])  # type: ignore
    df_pandas = pd.concat([zero_row, df_pandas]).reset_index(drop=True)  # type: ignore

    return df_pandas


def plot_qini(
    df: pd.DataFrame,
    x: str = "fraction",
    y: str = "ate",
    label: Optional[str] = None,
    plot_random: bool = True,
    bootstrap: bool = False,
    ax: Any = None,
) -> Any:
    """Plots the Qini curve.

    Args:
        df (pandas.DataFrame): a dataframe containing Qini estimates and cumulative population sizes or fractions.
            If bootstrap=True, df should also contain upper and lower bounds for the lift estimates.
        x (str, optional): column name in df that contains the cumulative population sizes or fractions.
            This defines the x-axis.
        y (str, optional): column name in df that contains the Qini estimates. This defines the y-axis.
        label (str, optional): name of the score estimation method to be shown on the legend. Defaults to None.
        plot_random (bool, optional): Indicator whether the random targeting line should be plotted. Defaults to True.
        bootstrap (bool, optional): indicates whether to use lower and upper bound values from df and plot a
            scatterplot with errorbars. If False, plots a barplot.
        ax (matplotlib.axes._subplots.AxesSubplot, optional): if specified, the plot will be plotted on this ax. Useful when creating a figure with subplots.

    Returns:
        (matplotlib.axes._subplots.AxesSubplot): the axis of the plot
    """

    df_qini = df.copy()

    ax = sns.lineplot(x=x, y=y, data=df_qini, label=label, ax=ax)

    if bootstrap:
        df_qini["lower"] = df_qini[y] - df_qini[f"{y}_lower"]
        df_qini["upper"] = df_qini[f"{y}_upper"] - df_qini[y]
        ax.errorbar(x=df_qini[x], y=df_qini[y], yerr=df_qini[["lower", "upper"]].values.T, fmt="ko")  # type: ignore

    xlabels = [f"{x:,.0f}" for x in ax.get_xticks() * 100]
    _ = ax.set_xticklabels(xlabels)
    ax.set(title="Qini curve", xlabel="% population targeted", ylabel="ATE over the whole population")

    if plot_random:
        # Fetch the overall ATE from the row with the highest % of targeted population
        df_qini = df_qini.sort_values(x)
        overall_ate = df_qini.iloc[-1][y].squeeze()
        plot_cate_random(overall_ate, ax=ax)

    return ax


def estimate_and_plot_qini(
    df: DataFrame,
    n_buckets: int = 30,
    score_colname: str = "cate_outcome",
    treatment_colname: str = "treatment",
    target_colname: str = "outcome",
    treatment_value: Union[str, int] = 1,
    control_value: Union[str, int] = 0,
    label: str = "Qini coefficient",
    plot_random: bool = True,
    bootstrap: bool = False,
    n_bootstraps: int = 100,
    ci_quantiles: Optional[List[float]] = None,
    ax: Any = None,
) -> Tuple:
    """Divides the data into buckets based on model score quantiles, estimates the Qini values
        (with or without confidence intervals), and plots them as a lineplot.

    Args:
        df (pyspark.sql.DataFrame): a dataframe containing the treatment indicators, the observed outcomes,
            and real-valued model scores
        n_buckets (int, optional): the number of quantiles to generate from the column <score_colname>
        score_colname (str, optional): the column name in df that contains the model scores
        treatment_colname (str, optional): the column name in df that contains the treatment indicators
        target_colname (str, optional): the column name in df that contains the target
        treatment_value (str or int, optional): the value in column <treatment_colname> that
            refers to the treatment group
        control_value (str or int, optional): the value in column <treatment_colname> that refers to the control group
        label (str, optional): name of the score estimation method to be shown on the legend. Defaults to None.
        plot_random (bool, optional): Indicator whether the random targeting line should be plotted. Defaults to True.
        bootstrap (bool, optional): if True, will perform bootstrapping and return confidence intervals
        n_bootstraps (int, optional): the number of bootstraps to perform. Only has an effect if bootstrap=True
        ci_quantiles (list of float, optional): the lower and upper confidence bounds.
            Only has an effect if bootstrap=True
        ax (matplotlib.axes._subplots.AxesSubplot, optional): if specified, the plot will be plotted on this ax. Useful when creating a figure with subplots.

    Returns:
        A tuple, containing:
        (pandas.DataFrame): a dataframe containing Qini estimates (with or without confidence intervals),
            cumulative population sizes and fractions
        (matplotlib.axes._subplots.AxesSubplot): the axis of the plot
    """

    if ci_quantiles is None:
        ci_quantiles = [0.025, 0.975]
    df_qini = estimate_qini(
        df=df,
        n_buckets=n_buckets,
        treatment_colname=treatment_colname,
        target_colname=target_colname,
        treatment_value=treatment_value,
        control_value=control_value,
        score_colname=score_colname,
        bootstrap=bootstrap,
        n_bootstraps=n_bootstraps,
        ci_quantiles=ci_quantiles,
    )

    qini_score = compute_qini_coefficient(df_qini)

    ax = plot_qini(
        df_qini, label=f"{label}: {round(qini_score, 4)}", plot_random=plot_random, bootstrap=bootstrap, ax=ax
    )

    return df_qini, ax


def plot_cate_random(overall_ate: float, ax: Any = None) -> Any:
    """Plots the random targeting line on a Qini plot.

    Args:
        overall_ate (float): the overall treatment effect across all instances, if all instances were targeted
        ax (matplotlib.axes._subplots.AxesSubplot, optional): if specified, the plot will be plotted on this ax. Useful when creating a figure with subplots.
    """

    x = np.linspace(0, 1, 10)
    y = x * overall_ate
    if ax:
        ax.plot(x, y, color="gray", linestyle="--")
    else:
        plt.plot(x, y, color="gray", linestyle="--")


def compute_auuc(df_qini: pd.DataFrame) -> float:
    """Computes the Area Under the Uplift Curve.

    Args:
        df_qini (pandas.DataFrame): a dataframe containing the Qini estimates
    Returns:
        A scalar representing the AUUC score
    """
    auuc = auc(df_qini["fraction"], df_qini["ate"])

    return auuc


def compute_qini_coefficient(df_qini: pd.DataFrame) -> float:
    """Computes the Qini coefficient

    Args:
        df_qini (pandas.DataFrame): a dataframe containing the Qini estimates
    Returns:
        A scalar representing the Qini coefficient
    """
    auuc = compute_auuc(df_qini)
    random_auc = auc(df_qini["fraction"].iloc[[0, -1]], df_qini["ate"].iloc[[0, -1]])

    qini_score = auuc - random_auc

    return qini_score


def estimate_cum_iroi(
    df: DataFrame,
    n_buckets: int = 30,
    score_colname: str = "cate_outcome",
    benefit_colname: str = "revenue",
    cost_colname: str = "cost",
    treatment_colname: str = "treatment",
    treatment_value: Union[str, int] = 1,
    control_value: Union[str, int] = 0,
    bootstrap: bool = False,
    n_bootstraps: int = 100,
    ci_quantiles: Optional[List[float]] = None,
) -> pd.DataFrame:
    """Divides the data into buckets based on model score quantiles and estimates cumulative iROI
        (with or without confidence intervals).

    Args:
        df (pyspark.sql.DataFrame): a dataframe containing the treatment indicators, the observed outcomes,
            and real-valued model scores
        n_buckets (int, optional): the number of quantiles to generate from the column <score_colname>
        score_colname (str, optional): the column name in df that contains the model scores
        benefit_colname (str, optional): the column name in df that contains the benefit
        cost_colname (str, optional): the column name in df that contains the cost
        treatment_colname (str, optional): the column name in df that contains the treatment indicators
        treatment_value (str or int, optional): the value in column <treatment_colname> that
            refers to the treatment group
        control_value (str or int, optional): the value in column <treatment_colname> that refers to the control group
        bootstrap (bool, optional): if True, will perform bootstrapping and return confidence intervals
        n_bootstraps (int, optional): the number of bootstraps to perform. Only has an effect if bootstrap=True
        ci_quantiles (list of float, optional): the lower and upper confidence bounds.
            Only has an effect if bootstrap=True

    Returns:
        (pandas.DataFrame): a dataframe containing cumulative iROI estimates (with or without confidence intervals),
            cumulative population sizes and fractions
    """

    # Assign quantile bucket to each row
    if ci_quantiles is None:
        ci_quantiles = [0.025, 0.975]
    bucket_colname = "bucket"
    df_buckets = _assign_bucket(
        df, n_buckets=n_buckets, score_colname=score_colname, bucket_colname="bucket", add_labels=False
    )

    # Estimate metrics, possibly bootstrap
    df_counts = _estimate_metrics(
        df_buckets,
        metric_type="iroi",
        treatment_colname=treatment_colname,
        benefit_colname=benefit_colname,
        cost_colname=cost_colname,
        treatment_value=treatment_value,
        control_value=control_value,
        cumulative=True,
        bucket_colname=bucket_colname,
        bootstrap=bootstrap,
        n_bootstraps=n_bootstraps,
    )

    if bootstrap:
        df_counts = _compute_ci(
            df_counts, bucket_colname=bucket_colname, relevant_cols=["iroi"], ci_quantiles=ci_quantiles
        )

    select_cols = [bucket_colname, "count", "fraction", "threshold"]
    select_cols += [col for col in df_counts.columns if col.startswith("iroi")]
    return df_counts.select(select_cols).orderBy(bucket_colname, ascending=False).toPandas()


def plot_cum_iroi(
    df: pd.DataFrame,
    x: str = "fraction",
    y: str = "iroi",
    label: Optional[str] = None,
    plot_overall: bool = True,
    bootstrap: bool = False,
    ax: Any = None,
) -> Any:
    """Plots the cumulative iROI curve.

    Args:
        df (pandas.DataFrame): a dataframe containing iROI estimates and cumulative population sizes or fractions.
            If bootstrap=True, df should also contain upper and lower bounds for the lift estimates.
        x (str, optional): column name in df that contains the cumulative population sizes or fractions.
            This defines the x-axis.
        y (str, optional): column name in df that contains the iROI estimates. This defines the y-axis.
        label (str, optional): name of the score estimation method to be shown on the legend. Defaults to None.
        plot_overall (bool, optional): Indicator whether the overall iROI line should be plotted. Defaults to True.
        bootstrap (bool, optional): indicates whether to use lower and upper bound values from df and plot a scatterplot
            with errorbars. If False, plots a barplot.
        ax (matplotlib.axes._subplots.AxesSubplot, optional): if specified, the plot will be plotted on this ax. Useful when creating a figure with subplots.

    Returns:
        (matplotlib.axes._subplots.AxesSubplot): the axis of the plot
    """

    df_iroi = df.copy()

    if bootstrap:
        df_iroi["lower"] = df_iroi[y] - df_iroi[f"{y}_lower"]
        df_iroi["upper"] = df_iroi[f"{y}_upper"] - df_iroi[y]
        if ax is None:
            ax = plt.gca()  # type: ignore
        ax.errorbar(x=df_iroi[x], y=df_iroi[y], yerr=df_iroi[["lower", "upper"]].values.T, label=label)  # type: ignore
    else:
        ax = sns.lineplot(x=x, y=y, data=df_iroi, label=label, ax=ax)

    if plot_overall:
        # Fetch the overall iROI from the row with the highest % of targeted population
        df_iroi = df_iroi.sort_values(x)
        overall_iroi = df_iroi.iloc[-1][y].squeeze()
        ax.axhline(overall_iroi, linestyle="dashed", color="gray")

    xlabels = [f"{x:,.0f}" for x in ax.get_xticks() * 100]
    _ = ax.set_xticklabels(xlabels)
    ax.set(title="iROI across % targeted", xlabel="% population targeted", ylabel="iROI")
    plt.legend()

    return ax


def estimate_and_plot_cum_iroi(
    df: DataFrame,
    n_buckets: int = 30,
    score_colname: str = "score",
    benefit_colname: str = "revenue",
    cost_colname: str = "cost",
    treatment_colname: str = "treatment",
    treatment_value: Union[int, str] = 1,
    control_value: Union[int, str] = 0,
    label: Optional[str] = None,
    plot_overall: bool = True,
    bootstrap: bool = False,
    n_bootstraps: int = 100,
    ci_quantiles: Optional[List[float]] = None,
    ax: Any = None,
) -> Tuple:
    """Divides the data into buckets based on model score quantiles, estimates cumulative iROI
        (with or without confidence intervals), and plots the estimates as a lineplot.

    Args:
        df (pyspark.sql.DataFrame): a dataframe containing the treatment indicators, the observed outcomes,
            and real-valued model scores
        n_buckets (int, optional): the number of quantiles to generate from the column <score_colname>
        score_colname (str, optional): the column name in df that contains the model scores
        benefit_colname (str, optional): the column name in df that contains the benefit
        cost_colname (str, optional): the column name in df that contains the cost
        treatment_colname (str, optional): the column name in df that contains the treatment indicators
        treatment_value (str or int, optional): the value in column <treatment_colname> that refers to the
            treatment group
        control_value (str or int, optional): the value in column <treatment_colname> that refers to the control group
        label (str, optional): name of the score estimation method to be shown on the legend. Defaults to None.
        plot_overall (bool, optional): Indicator the overall iROI line should be plotted. Defaults to True.
        bootstrap (bool, optional): if True, will perform bootstrapping and return confidence intervals
        n_bootstraps (int, optional): the number of bootstraps to perform. Only has an effect if bootstrap=True
        ci_quantiles (list of float, optional): the lower and upper confidence bounds.
            Only has an effect if bootstrap=True
        ax (matplotlib.axes._subplots.AxesSubplot, optional): if specified, the plot will be plotted on this ax. Useful when creating a figure with subplots.

    Returns:
        A tuple, containing:
        (pandas.DataFrame): a dataframe containing cumulative iROI estimates (with or without confidence intervals),
            cumulative population sizes and fractions
        (matplotlib.axes._subplots.AxesSubplot): the axis of the plot
    """

    if ci_quantiles is None:
        ci_quantiles = [0.025, 0.975]
    df_iroi = estimate_cum_iroi(
        df=df,
        n_buckets=n_buckets,
        treatment_colname=treatment_colname,
        benefit_colname=benefit_colname,
        cost_colname=cost_colname,
        treatment_value=treatment_value,
        control_value=control_value,
        score_colname=score_colname,
        bootstrap=bootstrap,
        n_bootstraps=n_bootstraps,
        ci_quantiles=ci_quantiles,
    )

    ax = plot_cum_iroi(df_iroi, label=label, plot_overall=plot_overall, bootstrap=bootstrap, ax=ax)

    return df_iroi, ax
