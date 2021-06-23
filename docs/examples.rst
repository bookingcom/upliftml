Examples
================================================

Training a T-learner in PySpark
--------------------------------------

.. code-block:: python

    from upliftml.models.pyspark import TLearnerEstimator
    from upliftml.evaluation import estimate_and_plot_qini
    from uplift.datasets import simulate_randomized_trial
    from pyspark.ml.classification import LogisticRegression


    # Read/generate the dataset and convert it to Spark if needed
    df_pd = simulate_randomized_trial(n=2000, p=6, sigma=1.0, binary_outcome=True)
    df_spark = spark.createDataFrame(df_pd)

    # Split the data into train, validation, and test sets
    df_train, df_val, df_test = df_spark.randomSplit([0.5, 0.25, 0.25])

    # Preprocess the datasets (for implementation of get_features_vector, see the full example notebook)
    num_features = [col for col in df_spark.columns if col.startswith('feature')]
    cat_features = []
    df_train_assembled = get_features_vector(df_train, num_features, cat_features)
    df_val_assembled = get_features_vector(df_val, num_features, cat_features)
    df_test_assembled = get_features_vector(df_test, num_features, cat_features)

    # Build a two-model estimator
    model = TLearnerEstimator(base_model_class=LogisticRegression,
                              base_model_params={'maxIter': 15},
                              predictors_colname='features',
                              target_colname='outcome',
                              treatment_colname='treatment',
                              treatment_value=1,
                              control_value=0)
    model.fit(df_train_assembled, df_val_assembled)

    # Apply the model to test data
    df_test_eval = model.predict(df_test_assembled)

    # Evaluate performance on the test set
    qini_values, ax = estimate_and_plot_qini(df_test_eval)

For complete examples with more estimators and evaluation functions, see the demo notebooks in the ``examples`` folder.