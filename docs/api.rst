API Design
================================================

Estimators
---------------

When designing the package, we've tried to keep the interface for all estimators and evaluation functions as close to each other as possible.
Both the PySpark-based and H2O-based estimators employ a sklearn-style interface with a ``fit`` and a ``predict`` function.

We've done our best to keep the parameters for initializing a model as similar as possible across the two backends (PySpark and H2O) and model types.
However, given that the inner workings of the different modeling techniques vary considerably, we've deviated from enforcing this goal in some cases.
For instance, while T-learner and S-learner require specifying only a single combination of ``base_model_class``, ``base_model_params``, and ``predictor_colnames``,
the X-learner, which proceeds in two stages, takes as input two sets of these parameters. Therefore, we strongly encourage the users to carefully check the
docstring of each estimator before use.

The PySpark-based and H2O-based estimators follow an almost identical interface. One exception being that the H2O-based estimators
require the ``predictor_colnames`` to be specified as a list of strings, while PySpark-based estimators instead require specifying
the ``predictors_colname`` as a single string. This ``predictors_colname`` is expected to represent an assembled vector from all features,
as is common in PySpark ML. The reason behind this design decision is to reduce the overhead of assembling the vector each time the user trains
a different estimator.
The other difference is that the ``predict`` functions returns a single column in case of H2O, while in PySpark it
returns the whole test dataframe with the predictions column attached to the dataframe. The reason for this is that while H2O returns
the predictions in the same order as the instances in the test dataframe, in PySpark it's impossible to recover the original
order within a single column and hence to ensure that the predictions would be aligned with the corresponding instances.

The following code block represents the basic signature of the estimators on the example of H2O-based T-learner.

.. code-block:: python

    class TLearnerEstimator

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
            output_colname: str = "cate_t",
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

        def fit(self, df_h2o_train: h2o.H2OFrame, df_h2o_val: h2o.H2OFrame = None) -> None:
        """Trains the T-learner.

        Args:
            df_h2o_train (h2o.H2OFrame): a dataframe containing the treatment indicators, the observed outcomes, and predictors
            df_h2o_val (h2o.H2OFrame, optional): a dataframe containing the treatment indicators, the observed outcomes, and predictors
        """

        def predict(self, df_h2o: h2o.H2OFrame) -> h2o.H2OFrame:
        """Applies the T-learner and returns treatment effect predictions.

        Args:
            df_h2o (h2o.H2OFrame): a dataframe containing predictors

        Returns:
            predictions (h2o.H2OFrame): a single column containing treatment effect predictions
        """

Evaluation
---------------

The evaluation module is common for all estimators and works on PySpark dataframes. This means that the user is expected to convert
their evaluation dataset back to PySpark when using H2O estimators.

All the different types of evaluation curves follow a similar structure. There exists:
1) a function for estimating the values on the curve, which takes as input a PySpark dataframe and returns a Pandas dataframe,
2) a function for plotting the curve, which takes as input the Pandas dataframe from the estimation function and returns the axis of the plot,
3) a function that combines the two steps, i.e., calls both the estimation and plotting functions.

The following code block represents the basic signature of these three types of functions on the example of CATE per bucket.

.. code-block:: python

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