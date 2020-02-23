
# coding: utf-8
"""
Bayesian workflow with Python and Stan using air quality measurements 
from Leppävaara air quality monitoring station in Espoo. Testing three 
different model setups.

Input data: Hourly PM2.5 concentration measurerements (ug/m3).

@author: hannahannuniemi
"""


import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import pystan
import psis
import stan_utility
from scipy.stats import norm
from scipy import stats
import plot_tools
import warnings


# Read csv data
df = pd.read_csv('leppavaara_air_quality_2013-2018.csv')
df_pos = df[df['Particulate matter < 2.5 µm (ug/m3)']>0]
df_pos.head()


# Monthly averages
gm = df_pos.groupby(['Year', 'm'])
monthly_averages = gm.aggregate({'Particulate matter < 2.5 µm (ug/m3)':np.mean})
monthly = monthly_averages.values


# Model 1: Gaussian linear model with standardized data for all months.
""" 
    Model used from demo examples:
    https://github.com/avehtari/BDA_py_demos/blob/master/demos_pystan/lin_std.stan
"""

model = """
data {
  int<lower=0> N; // number of data points
  vector[N] x; //
  vector[N] y; //
  real xpred; // input location for prediction
}
transformed data {
  vector[N] x_std;
  vector[N] y_std;
  real xpred_std;
  x_std = (x - mean(x)) / sd(x);
  y_std = (y - mean(y)) / sd(y);
  xpred_std = (xpred - mean(x)) / sd(x);
}
parameters {
  real alpha;
  real beta;
  real<lower=0> sigma_std;
}
transformed parameters {
  vector[N] mu_std;
  mu_std = alpha + beta*x_std;
}
model {
  alpha ~ normal(0, 1);
  beta ~ normal(0, 1);
  y_std ~ normal(mu_std, sigma_std);
}
generated quantities {
  vector[N] mu;
  real<lower=0> sigma;
  real ypred;
  vector[N] log_lik;
  real y_rep[N];
  mu = mu_std*sd(y) + mean(y);
  sigma = sigma_std*sd(y);
  ypred = normal_rng((alpha + beta*xpred_std)*sd(y)+mean(y), sigma*sd(y));
  y_rep = normal_rng((alpha + beta*x_std)*sd(y)+mean(y), sigma*sd(y));
  for (i in 1:N)
    log_lik[i] = normal_lpdf(y[i] | mu[i], sigma);
}
"""
# Model parameters
N=len(monthly)
x=np.arange(1,71)
y=monthly[:,0]
xpred=71

data = {
             'N': N,
             'x': x,
             'y': y,
             'xpred': xpred
}

model_1 = pystan.StanModel(model_code=model)
fit1 = model_1.sampling(data=data)
samples1 = fit1.extract(permuted=True)
y_rep1 = fit1.extract(permuted=True)['y_rep']

# Model convergence diagnostics
stan_utility.check_treedepth(fit1)
stan_utility.check_energy(fit1)
stan_utility.check_div(fit1)

# Results for model 1
print('Pr(beta < 0) = {}'.format(np.mean(samples1['beta'] < 0)))

# plotting results
figsize = plt.rcParams['figure.figsize'].copy()
figsize[0] *= 1.4
figsize[1] *= 2.5
fig, axes = plt.subplots(3, 1, figsize=figsize)

color_scatter = 'C0'  
color_line = 'C1'     
color_shade = (
    1 - 0.1*(1 - np.array(mpl.colors.to_rgb(color_line)))
)

ax = axes[0]
ax.fill_between(
    x,
    np.percentile(samples1['mu'], 5, axis=0),
    np.percentile(samples1['mu'], 95, axis=0),
    color=color_shade
)
ax.plot(
    x,
    np.percentile(samples1['mu'], 50, axis=0),
    color=color_line,
    linewidth=1
)
ax.scatter(x, y, 5, color=color_scatter)
ax.set_xlabel('Month number from Jan 2013')
ax.set_ylabel('Monthly average PM2.5 concentration (ug/m3)')
ax.set_xlim((0, 72))

# plot 2: histogram
ax = axes[1]
ax.hist(samples1['beta'], 50)
ax.set_xlabel('beta')

# plot 3: histogram
ax = axes[2]
ax.hist(samples1['ypred'], 50)
ax.set_xlabel('y-prediction for November 2018')
fig.tight_layout();


# Model 2: Gaussian linear model with standardized data only for November
data_november = df_pos[df_pos.m==11]
data_november = data_november.dropna()
data_november.head()

# Monthly averages
gm = data_november.groupby(['Year', 'm'])
monthly_averages_nv = gm.aggregate({'Particulate matter < 2.5 µm (ug/m3)':np.mean})
monthly_nv = monthly_averages_nv.values

# Model 2 parameters
N=len(monthly_nv)
x=np.arange(1,6)
y=monthly_nv[:,0]
xpred=6

data_nv = {
             'N': N,
             'x': x,
             'y': y,
             'xpred': xpred
}

model_2 = pystan.StanModel(model_code=model)
fit2 = model_2.sampling(data=data_nv)
samples2 = fit2.extract(permuted=True)

# Model convergence diagnostics
stan_utility.check_treedepth(fit2)
stan_utility.check_energy(fit2)
stan_utility.check_div(fit2)

# Results for model 2
print('Pr(beta < 0) = {}'.format(np.mean(samples2['beta'] < 0)))

# plotting results
figsize = plt.rcParams['figure.figsize'].copy()
figsize[0] *= 1.4
figsize[1] *= 2.5
fig, axes = plt.subplots(3, 1, figsize=figsize)

color_scatter = 'C0'  
color_line = 'C1'     
color_shade = (
    1 - 0.1*(1 - np.array(mpl.colors.to_rgb(color_line)))
)

ax = axes[0]
ax.fill_between(
    x,
    np.percentile(samples2['mu'], 5, axis=0),
    np.percentile(samples2['mu'], 95, axis=0),
    color=color_shade
)
ax.plot(
    x,
    np.percentile(samples2['mu'], 50, axis=0),
    color=color_line,
    linewidth=1
)
ax.scatter(x, y, 5, color=color_scatter)
ax.set_xlabel('November 2013 to 2017')
ax.set_ylabel('Monthly average PM2.5 concentration (ug/m3)')
ax.set_xlim((0, 7))

# plot 2: histogram
ax = axes[1]
ax.hist(samples2['beta'], 50)
ax.set_xlabel('beta')

# plot 3: histogram
ax = axes[2]
ax.hist(samples2['ypred'], 50)
ax.set_xlabel('y-prediction for x={}'.format(xpred))
fig.tight_layout();


# Model 3: Hierarchical model with common variance
m_av = df_pos.groupby(['Year', 'm']).mean().reset_index()
m_values = m_av['Particulate matter < 2.5 µm (ug/m3)'].values
xx = np.tile(np.arange(1, 13), 6) 
x = xx[0:70]
N = len(x)

data_hier = {
             'K': 12,
             'N': N,
             'x': x,
             'y': m_values,
}

model_hier = """
         data {
             int<lower=0> N; // number of data points
             int<lower=0> K; // number of groups
             int<lower=1, upper=K> x[N]; // group indicator
             vector[N] y;
         }
         parameters {
             real mu0;
             real<lower=0> sigma0;
             vector[K] mu; // group means
             real<lower=0> sigma; // common std
         }
         model {
             mu0 ~ normal(10,7);      // weakly informative prior
             sigma0 ~ cauchy(0,4);     // weakly informative prior
             sigma ~ cauchy(0,4);      // weakly informative prior
             mu ~ normal(mu0, sigma0);
             y ~ normal(mu[x], sigma);
         }
         generated quantities {
             real ypred_11;
             vector[N] log_lik;
             real y_rep[N];
             ypred_11 = normal_rng(mu[11], sigma);
             y_rep = normal_rng(mu[x], sigma);
             for (i in 1:N)
                 log_lik[i] = normal_lpdf(y[i] | mu[x[i]], sigma);
         }
"""
# Fitting hierarchical model
model = pystan.StanModel(model_code=model_hier)
hier_fit = model.sampling(data=data_hier, seed=1, control=dict(adapt_delta=0.97))
mu = hier_fit.extract(permuted=True)['mu']
y_rep_hier = hier_fit.extract(permuted=True)['y_rep']

# Model convergence diagnostics
stan_utility.check_treedepth(hier_fit)
stan_utility.check_energy(hier_fit)
stan_utility.check_div(hier_fit)

# Results for model 3
# plot violins
plt.violinplot(mu, showmeans=True, showextrema=False)
plt.xticks(np.arange(1, 13))
plt.xlabel('month')
plt.ylabel('mu');

# Model comparison with LOO, calculated PSIS-LOO values
# Psis.py Copyright 2017 Aki Vehtari, Tuomas Sivula
loo1, loos1, k1 = psis.psisloo(fit1['log_lik'])
loo2, loos2, k2 = psis.psisloo(fit2['log_lik'])
loo3, loos3, k3 = psis.psisloo(hier_fit['log_lik'])

# plot the pareto k-values
figsize = plt.rcParams['figure.figsize'].copy()
figsize[0] *= 2.5  # width
figsize[1] *= 1.5  # height
fig, axes = plt.subplots(1, 3, figsize=figsize)

ax = axes[0]
ax.scatter(np.arange(1,71,1), k1)
ax.set_ylabel('pareto k')
ax.set_xlabel('observation left out')
ax.set_ylim((-0.2, 1.0))
ax.axhline(y=0.7, linestyle='--', color='r')
ax.axhline(y=0.5, linestyle='--', color='g')
ax.set_title('model 1')

ax = axes[1]
ax.scatter(np.arange(1,6,1), k2)
ax.set_ylabel('pareto k')
ax.set_xlabel('observation left out')
ax.set_ylim((-0.2, 1.0))
ax.axhline(y=0.7, linestyle='--', color='r')
ax.axhline(y=0.5, linestyle='--', color='g')
ax.set_title('model 2')

ax = axes[2]
ax.scatter(np.arange(1,71,1), k3)
ax.set_ylabel('pareto k')
ax.set_xlabel('observation left out')
ax.set_ylim((-0.2, 1.0))
ax.axhline(y=0.7, linestyle='--', color='r')
ax.axhline(y=0.5, linestyle='--', color='g')
ax.set_title('model 3')


# posterior predictive checking for model_1
# code adapted from notebook "Towards A Principled Bayesian Workflow" by Michael Betancourt,
# https://github.com/betanalpha/jupyter_case_studies/blob/master/principled_bayesian_workflow/principled_bayesian_workflow.ipynb

light="#DCBCBC"
light_highlight="#C79999"
mid="#B97C7C"
mid_highlight="#A25050"
dark="#8F2727"
dark_highlight="#7C0000"
green="#00FF00"

max_y = 20
B = max_y + 1

bins = [b - 0.5 for b in range(B + 1)]

idxs = [ idx for idx in range(B) for r in range(2) ]
xs = [ idx + delta for idx in range(B) for delta in [-0.5, 0.5]]
      
obs_counts = np.histogram(monthly, bins=bins)[0]
pad_obs_counts = [ obs_counts[idx] for idx in idxs ]

counts = [np.histogram(y_rep1[n], bins=bins)[0] for n in range(4000)]
probs = [10, 20, 30, 40, 50, 60, 70, 80, 90]
creds = [np.percentile([count[b] for count in counts], probs)
         for b in range(B)]
pad_creds = [ creds[idx] for idx in idxs ]

plt.fill_between(xs, [c[0] for c in pad_creds], [c[8] for c in pad_creds],
                  facecolor=light, color=light)
plt.fill_between(xs, [c[1] for c in pad_creds], [c[7] for c in pad_creds],
                  facecolor=light_highlight, color=light_highlight)
plt.fill_between(xs, [c[2] for c in pad_creds], [c[6] for c in pad_creds],
                  facecolor=mid, color=mid)
plt.fill_between(xs, [c[3] for c in pad_creds], [c[5] for c in pad_creds],
                  facecolor=mid_highlight, color=mid_highlight)
plt.plot(xs, [c[4] for c in pad_creds], color=dark)

plt.plot(xs, pad_obs_counts, linewidth=2.5, color="white")
plt.plot(xs, pad_obs_counts, linewidth=2.0, color="black")

plt.gca().set_xlabel("y")
plt.gca().set_ylabel("Posterior Predictive Distribution")
plt.gca().set_title("Model_1: All data")
plt.show()

