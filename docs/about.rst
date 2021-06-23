About UpliftML
===================

**UpliftML** is a Python package for scalable uplift modeling from experimental data. To accommodate working with big data, the package uses PySpark and H2O models as base learners for the uplift models. Evaluation functions expect a PySpark dataframe as input.

**Uplift modeling** is a collection of machine learning techniques for estimating causal effects of a
treatment at the individual or subgroup level. In particular, these techniques estimate the Conditional Average Treatment Effect (CATE) given available pre-treatment information about the individual.

Over the last years, causality and uplift modeling have become key trends
in personalization at online e-commerce platforms, enabling to select the best treatment for each user in order to maximize
the target business metric. Beyond e-commerce, uplift modeling has been used for clinical treatment personalization and in various social science applications, e.g., for selecting the best voting mobilization strategies for individuals.

Uplift modeling can be particularly useful for personalized promotional campaigns, where the the goal is to select which users to target in order to maximize the overall uplift without exceeding a specified **budget or ROI constraint**.
In those cases, estimating CATE alone is not sufficient to make optimal decisions. Instead, the potential benefit caused by a promotion needs to be weighed against the potential costs, giving rise to **constrained uplift modeling** methods.

The **UpliftML** library includes PySpark and H2O implementations for the following:

- 6 metalearner approaches for CATE estimation: T-learner :cite:p:`kunzel2019metalearners`, S-learner :cite:p:`kunzel2019metalearners`, X-learner :cite:p:`kunzel2019metalearners`, R-learner :cite:p:`nie2021quasi`, Class Variable Transformation :cite:p:`jaskowski2012uplift`, Transformed Outcome approach :cite:p:`athey2015machine`.
- The Retrospective Estimation :cite:p:`goldenberg2020free` technique for constrained uplift modeling.
- Uplift and iROI-based evaluation and plotting functions with bootstrapped confidence intervals. Currently implemented: ATE, ROI, iROI, CATE per category/quantile, CATE lift, Qini/AUUC curves :cite:p:`radcliffe2011real`, Qini/AUUC score :cite:p:`radcliffe2011real`, cumulative iROI curves.
