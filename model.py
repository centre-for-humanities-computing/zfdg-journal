"""Bayesian DirichletMultinomial model of book ratings. :D"""
import ast
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

# Loading the data from disk
data = pd.read_excel("data.xlsx")


def dict_to_array(d: dict) -> np.ndarray:
    """Turns the dictionary of ratings into an array with the counts
    at each position."""
    a = np.empty(5)
    for rating in range(1, 5 + 1):
        a[rating - 1] = d.get(str(rating), 0)
    return a


# Here I gather the rating distributions into a matrix
rate_dists = np.stack(
    data["RATING_DIST_DICT"]
    .map(ast.literal_eval)  # Parse dict from string
    .map(dict_to_array)  # Turns dicts into arrays
)
# I gather excellence scores to a vector
excellence = np.array(data.EXCELLENCE_ALL)

# Model description with PyMC
n_obs = excellence.shape[0]
with pm.Model(
    coords={
        "excellence": ["non-excellent", "excellent"],
        "ratings": np.arange(1, 6),
    }
) as dirichlet_model:
    # Hyperprior for concentration
    scale = pm.Gamma("scale", 1, 1, dims=("excellence"))
    shape = pm.Gamma("shape", 1, 1, dims=("excellence"))
    # Concentration on the rating level
    concentration = pm.Gamma(
        "concentration", shape, scale, dims=("ratings", "excellence")
    ).T
    # Observations are generated from a DirichletMultinomial
    obs = pm.DirichletMultinomial(
        "rating",
        rate_dists.sum(axis=1),
        concentration[excellence],
        observed=rate_dists,
    )
dirichlet_model.debug()

Path("results").mkdir(exist_ok=True)
with dirichlet_model:
    # We sample the posterior
    idata = pm.sample()
    # We sample the posterior predictive
    idata.extend(pm.sample_posterior_predictive(idata))
    # Savin' that juicy data
    idata.to_netcdf("results/inference_data.netcdf")


Path("figures").mkdir(exist_ok=True)
# Plot the effect in scale
az.plot_forest(idata, var_names="scale")
plt.tight_layout()
plt.savefig("figures/scale.png")

# Plot the effect in shape
az.plot_forest(idata, var_names="shape")
plt.tight_layout()
plt.savefig("figures/shape.png")

# Plot the effect in concentration
az.plot_forest(idata, var_names="concentration")
plt.tight_layout()
plt.savefig("figures/concentration.png")


# Plot posterior predictive
az.plot_ppc(idata, group="posterior")
plt.tight_layout()
plt.savefig("figures/posterior_predictive.png")

# Plot trace to inspect if shit worked
az.plot_trace(idata)
plt.tight_layout()
plt.savefig("figures/trace_plot.png")

# Save summary
summary = az.summary(idata)
summary.to_csv("results/summary.csv")
