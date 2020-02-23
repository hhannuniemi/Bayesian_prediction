# Bayesian_prediction

Aim of this project is to practise Bayesian workflow with Python and Stan using air quality measurements from air quality monitoring station. This study assses the fine particle (PM2.5) concentration in past five years and explore if the air quality is getting better, and if there is a difference in fine particle concentrations between months. The past data is used to predict the average fine particle concentration in next month.

Three models are tested: 
1. Gaussian linear model with standardized data for all months,
2. Same as first model, but using only data from November months,
3. Hierarchical model with hierarchical prior for mean and common variance.

For model comparison Pareto-smoothed importance sampling (PSIS) is used to calculate leave-one-out cross validation.
