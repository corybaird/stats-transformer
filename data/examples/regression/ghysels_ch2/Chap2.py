import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import jarque_bera
from scipy.stats import f
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri

# Load the simulated data
sim_data_full = pd.read_csv('simulated_datac2.csv')

# Define the estimation and forecasting samples
sim_data_est = sim_data_full.iloc[101:301]  # estimation sample
sim_data_fore = sim_data_full.iloc[301:401]  # forecasting sample

# Fit the OLS model
ols_fit = smf.ols(formula='y ~ x', data=sim_data_est).fit()
print(ols_fit.summary())

# Breusch-Pagan Test
bp_test = het_breuschpagan(ols_fit.resid, ols_fit.model.exog)
print('Breusch-Pagan test statistic:', bp_test[0])
print('Breusch-Pagan test p-value:', bp_test[1])

# Add residuals and their squares to the estimation data
sim_data_est['eps'] = ols_fit.resid
sim_data_est['x2'] = sim_data_est['x'] ** 2
sim_data_est['eps2'] = sim_data_est['eps'] ** 2

# Jarque-Bera Test for normality
jb_test = jarque_bera(sim_data_est['eps'])
print('Jarque-Bera test statistic:', jb_test[0])
print('Jarque-Bera test p-value:', jb_test[1])

# Histogram of residuals
plt.hist(sim_data_est['eps'], bins=20, edgecolor='k')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Histogram of Residuals')
plt.show()

# White Test
white_fit = smf.ols(formula='eps2 ~ x2 + x', data=sim_data_est).fit()
print(white_fit.summary())

# Durbin-Watson Test
dw_test = durbin_watson(ols_fit.resid)
print('Durbin-Watson test statistic:', dw_test)

# Function to perform Chow Test
def chow_test(data, formula, split_point):
    data1 = data.iloc[:split_point]
    data2 = data.iloc[split_point:]
    
    ols1 = smf.ols(formula=formula, data=data1).fit()
    ols2 = smf.ols(formula=formula, data=data2).fit()
    ols_full = smf.ols(formula=formula, data=data).fit()
    
    rss1 = ols1.ssr
    rss2 = ols2.ssr
    rss_full = ols_full.ssr
    
    k = ols_full.df_model + 1  # number of parameters
    n1 = len(data1)
    n2 = len(data2)
    
    chow_stat = ((rss_full - (rss1 + rss2)) / k) / ((rss1 + rss2) / (n1 + n2 - 2 * k))
    p_value = 1 - f.cdf(chow_stat, k, n1 + n2 - 2 * k)
    
    return chow_stat, p_value

# Perform Chow Test for specified ranges
for (start, end) in [(0.1, 0.25), (0.25, 0.5), (0.5, 0.75)]:
    split_point = int(start * len(sim_data_est))
    chow_stat, p_value = chow_test(sim_data_est, 'y ~ x', split_point)
    print(f'Chow test ({start*100:.0f}% - {end*100:.0f}%) p-value:', p_value)

# Bai-Perron test
# Assign the data to an R environment
robjects.globalenv['sim_data_est'] = r_sim_data_est

# Run the breakpoints function in R
r_breakpoints = robjects.r('breakpoints(y ~ x, data = sim_data_est, h = 0.15)')
breakpoints = list(r_breakpoints.rx2('breakpoints'))
print("Breakpoints:", breakpoints)

# Dummy variable for the breakpoints
# Fit the model with the dummy variable and interaction
olsD_fit = smf.ols('y ~ D + x + D:x', data=sim_data_est).fit()
print(olsD_fit.summary())

# Simple forecasts - with / no dummy
yhat = {}
yhat['y'] = ols_fit.predict(sim_data_fore)  # no dummy
yhat['yD'] = olsD_fit.predict(sim_data_fore)  # with dummy

# Calculate the standard error of the predictions for the model with the dummy
yD_se = np.sqrt(np.sum(olsD_fit.resid ** 2) / 198)
yhat['yD.se'] = yD_se

# Calculate the upper and lower bounds of the confidence interval
yhat['yD.up'] = yhat['yD'] + 1.96 * yD_se
yhat['yD.low'] = yhat['yD'] - 1.96 * yD_se
