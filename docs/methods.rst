Uplift Modeling
================================================
**Uplift modeling**  aims to estimate the Conditional Average Treatment Effect (CATE) from experimental or observational data.
In particular, we are interested in estimating the causal effect of a treatment :math:`T` on the outcome :math:`Y` of an individual characterized by features :math:`X`.
Given a binary treatment :math:`T`,

.. math::
    CATE(x) = E[Y(1) - Y(0) | X=x]

In unconfounded data (ideally collected via a randomized controlled trial), this is equivalent to estimating:

.. math::
    E[Y | T=1, X=x] - E[Y | T=0, X=x]

**Constrained uplift modeling** can be formulated as a binary Knapsack problem as follows:

.. math::
    max ~& \sum_i{CATE_Y(x_i)} \cdot z_i \\
    s.t. ~& \sum_i{CATE_C(x_i)} \cdot z_i \leq B,

where :math:`CATE_Y(x_i)` represents the incremental benefit caused by the treatment, :math:`CATE_C(x_i)` the incremental costs, :math:`z_i\in\{0,1\}` is the treatment allocation indicator and `B` is the available global budget.

Some possible solutions to the constrained problem above include Fractional Approximation :cite:p:`goldenberg2020free`, Retrospective Estimation :cite:p:`goldenberg2020free`, label adjustment based on Lagrangian Subgradient method :cite:p:`du2019improve`, and instrumental variable approaches :cite:p:`sun2021treatment`.

The following sections give an overview of the approaches implemented in UpliftML.

CATE Estimators
--------------------------------
**T-learner** :cite:p:`kunzel2019metalearners` estimates CATE as the difference in estimates from two separate models: :math:`E[Y | T=1, X] - E[Y | T=0, X]`.

**S-learner** :cite:p:`kunzel2019metalearners` estimates CATE by training a single model for :math:`E[Y | T, X]`, applying the model with :math:`T=1` and :math:`T=0` and using the difference in these estimates as the estimated treatment effect.

**X-learner** :cite:p:`kunzel2019metalearners` estimates CATE in three stages:

    1. Train a T-learner to get scores :math:`\hat{Y}_1` and :math:`\hat{Y}_0`.
    2. Train regression models to predict the residuals: :math:`\tau_1 = E[Y(1) - \hat{Y}_1) | X]` and :math:`\tau_0 = E[\hat{Y}_0 - Y(0) | X]`.
    3. Estimate the treatment effect as a weighted average: :math:`\tau_(X) = p(X) \cdot \tau_0(X) + [1 - p(X)] \cdot \tau_1(X)`, where :math:`p(X)` is often chosen to be the propensity score, i.e., :math:`p(X) = E[T=1 | X]`.

**R-learner** :cite:p:`nie2021quasi` estimates CATE in two stages:

    1. Using cross-fitting, on the training set train a marginal target estimator to get scores :math:`\hat{Y}` and a propensity estimator to get scores :math:`\hat{T}`, calculate the residuals on the validation set.
    2. Train a final estimator on the residuals: :math:`\tau_(X) = E[(Y- \hat{Y}) / (T - \hat{T}) | X]` using :math:`(T - \hat{T})^2` as weights.


**Class Variable Transformation** (CVT) :cite:p:`jaskowski2012uplift` estimates CATE by transforming the target variable into a new target variable :math:`Z`, such that the treatment effect :math:`\tau(X) = 2 \cdot E[Z | X] - 1`.
This transformation results in a classification problem and is, thus, slightly different from the TransformedOutcomeEstimator, which results in a regression problem.
Can only be used on randomized controlled trial (RCT) data with a 50-50 split into treatment and control groups.


**Transformed Outcome** approach :cite:p:`athey2015machine` estimates CATE by transforming the outcome, such that the expectation of the transformed outcome corresponds to the treatment effect.
This transformation results in a regression problem and is, thus, slightly different from the CVT approach, which results in a classification problem.


Estimators for Constrained Uplift Modeling
---------------------------------------------
**Retrospective Estimation** :cite:p:`goldenberg2020free` estimates :math:`E[T | Y=1, X]`, which corresponds to estimating the multiplicative treatment effect :math:`E[Y | T=1, X] / E[Y | T=0, X]` in case of RCT data with a 50-50 split into treatment and control groups.

