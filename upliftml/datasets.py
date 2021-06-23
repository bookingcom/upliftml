import numpy as np
import pandas as pd
from scipy.stats import gamma  # type: ignore


def _sigmoid(x: float) -> float:
    """Helper function to apply sigmoid to a float.
    Args:
        x: a float to apply the sigmoid function to.

    Returns:
        (float): x after applying the sigmoid function.
    """
    return 1 / (1 + np.exp(-x))


def simulate_randomized_trial(
    n: int = 1000, p: int = 5, sigma: float = 1.0, binary_outcome: bool = False, add_cost_benefit: bool = False
) -> pd.DataFrame:
    """Simulates a synthetic dataset corresponding to a randomized trial
        The version with continuous outcome and without cost/benefit columns corresponds to Setup B in Nie X. and Wager S. (2018) 'Quasi-Oracle Estimation of Heterogeneous Treatment Effects' and is aligned with the implementation in the CausalML package.
    Args:
        n (int, optional): number of observations to generate
        p (int optional): number of covariates. Should be >= 5, since treatment heterogeneity is determined based on the first 5 features.
        sigma (float): standard deviation of the error term
        binary_outcome (bool): whether the outcome should be binary or continuous
        add_cost_benefit (bool): whether to generate cost and benefit columns
    Returns:
        (pandas.DataFrame): a dataframe containing the following columns:
            - treatment
            - outcome
            - propensity
            - expected_outcome
            - actual_cate
            - benefit (only if add_cost_benefit=True)
            - cost (only if add_cost_benefit=True)
    """

    X = np.random.normal(loc=0.0, scale=1.0, size=n * p).reshape((n, -1))
    b = np.maximum(np.repeat(0.0, n), X[:, 0] + X[:, 1] + X[:, 2]) + np.maximum(np.repeat(0.0, n), X[:, 3] + X[:, 4])
    e = np.repeat(0.5, n)
    tau = X[:, 0] + np.log1p(np.exp(X[:, 1]))

    w = np.random.binomial(1, e, size=n)

    if binary_outcome:
        y1 = b + (1 - 0.5) * tau + sigma * np.random.normal(loc=0.0, scale=1.0, size=n)
        y0 = b + (0 - 0.5) * tau + sigma * np.random.normal(loc=0.0, scale=1.0, size=n)
        y1_binary = pd.Series(_sigmoid(y1) > 0.5).astype(np.int32)  # potential outcome when w=1
        y0_binary = pd.Series(_sigmoid(y0) > 0.5).astype(np.int32)  # potential outcome when w=0

        # observed outcome
        y = y0_binary
        y[w == 1] = y1_binary  # type: ignore

        # ensure that tau is between [-1, 1]
        tau = _sigmoid(y1) - _sigmoid(y0)

    else:
        y = b + (w - 0.5) * tau + sigma * np.random.normal(loc=0.0, scale=1.0, size=n)

    data = pd.DataFrame({"treatment": w, "outcome": y, "propensity": e, "expected_outcome": b, "actual_cate": tau})
    features = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(1, X.shape[1] + 1)])
    data = pd.concat([data, features], axis=1)

    if add_cost_benefit:
        data["benefit"] = gamma.rvs(3, size=n)
        data.loc[data["outcome"] == 0, "benefit"] = 0
        data["cost"] = data["benefit"] * 0.25
        data.loc[data["treatment"] == 0, "cost"] = 0

    return data
