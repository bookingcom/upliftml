# UpliftML: A Python Package for Scalable Uplift Modeling
**UpliftML** is a Python package for scalable unconstrained and constrained uplift modeling from experimental data. To accommodate working with big data, the package uses PySpark and H2O models as base learners for the uplift models. Evaluation functions expect a PySpark dataframe as input.

**Uplift modeling** is a family of techniques for estimating the Conditional Average Treatment Effect (CATE) from experimental or observational data using machine learning. In particular, we are interested in estimating the causal effect of a treatment T on the outcome Y of an individual characterized by features X. In experimental data with binary treatments and binary outcomes, this is equivalent to estimating Pr(Y=1 | T=1, X=x) - Pr(Y=1 | T=0, X=x).

In many practical use cases the goal is to select which users to target in order to maximize the overall uplift without exceeding a specified **budget or ROI constraint**. In those cases, estimating uplift alone is not sufficient to make optimal decisions and we need to take into account the costs and monetary benefit incurred by the treatment.

Uplift modeling is an emerging tool for various personalization applications. Example use cases include marketing campaigns personalization and optimization, personalized pricing in e-commerce, and clinical treatment personalization.

The **UpliftML** library includes PySpark/H2O implementations for the following:
- 6 metalearner approaches for uplift modeling: T-learner[1], S-learner[1], X-learner[1], R-learner[2], class variable transformation[3], transformed outcome approach[4].
- The Retrospective Estimation[5] technique for uplift modeling under ROI constraints.
- Uplift and iROI-based evaluation and plotting functions with bootstrapped confidence intervals. Currently implemented: ATE, ROI, iROI, CATE per category/quantile, CATE lift, Qini/AUUC curves[6], Qini/AUUC score[6], cumulative iROI curves.

For detailed information about the package, read the [UpliftML documentation](https://upliftml.readthedocs.io/).

# Installation
Install the latest release from PyPI:

```
$ pip install upliftml
```

# Quick Start

```python
from upliftml.models.pyspark import TLearnerEstimator
from upliftml.evaluation import estimate_and_plot_qini
from upliftml.datasets import simulate_randomized_trial
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
```

For complete examples with more estimators and evaluation functions, see the demo notebooks in the ``examples`` folder.

# Contributing
If interested in contributing to the package, get started by reading our [contributor guidelines](CONTRIBUTING.md).

# License
The project is licensed under [Apache 2.0 License](https://github.com/bookingcom/upliftml/blob/main/LICENSE)

# Citation
If you use UpliftML, please cite it as follows:

Irene Teinemaa, Javier Albert, Nam Pham. **UpliftML: A Python Package for Scalable Uplift Modeling.** https://github.com/bookingcom/upliftml, 2021. Version 0.0.1.

```
@misc{upliftml,
  author={Irene Teinemaa, Javier Albert, Nam Pham},
  title={{UpliftML}: {A Python Package for Scalable Uplift Modeling}},
  howpublished={https://github.com/bookingcom/upliftml},
  note={Version 0.0.1},
  year={2021}
}
```


# Resources
Documentation:
* [UpliftML documentation](https://upliftml.readthedocs.io/)

Tutorials and blog posts:
* [Retrospective Estimation (blog post)](https://booking.ai/free-lunch-40a963e12b0a)
* [Uplift modeling tutorial at WebConf'2021](https://booking.ai/uplift-modeling-f9759e3fb51e)
* [Personalization in Practice tutorial at WSDM'2021](https://booking.ai/personalization-in-practice-2bb4bc680eb3)

Related packages:
* [CausalML](https://github.com/uber/causalml): a Python package for uplift modeling and causal inference with machine learning
* [EconML](https://github.com/microsoft/EconML): a Python package for estimating heterogeneous treatment effects from observational data via machine learning

# References

1. Sören R. Künzel, Jasjeet S. Sekhon, Peter J. Bickel, and Bin Yu. Metalearners for estimating heterogeneous treatment effects using machine learning. Proceedings of the National Academy of Sciences, 2019.
2. Xinkun Nie and Stefan Wager. Quasi-oracle estimation of heterogeneous treatment effects. arXiv preprint arXiv:1712.04912, 2017.
3. Maciej Jaskowski and Szymon Jaroszewicz. Uplift modeling for clinical trial data. ICML Workshop on Clinical Data Analysis, 2012.
4. Susan Athey and Guido W. Imbens. Machine learning methods for estimating heterogeneous causal effects. stat, 1050(5), 2015.
5. Dmitri Goldenberg, Javier Albert, Lucas Bernardi, Pablo Estevez Castillo. Free Lunch! Retrospective Uplift Modeling for Dynamic Promotions Recommendation within ROI Constraints. In Fourteenth ACM Conference on Recommender Systems (pp. 486-491), 2020.
6. Nicholas J Radcliffe and Patrick D Surry. Real-world uplift modelling with significance based uplift trees. White Paper tr-2011-1, Stochastic Solutions, 2011.
