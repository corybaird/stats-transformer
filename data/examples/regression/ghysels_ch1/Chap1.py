import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.eval_measures import rmse, meanabs

### 1.11.1 Data Simulation Procedure ###
# Load the data
sim_data = pd.read_csv('simulated_data.csv')
sim_data_est = sim_data.iloc[101:301]  # estimation sample
sim_data_fore = sim_data.iloc[301:401]  # forecasting sample

# Plot Y
plt.figure(figsize=(10, 6))
sns.lineplot(x=range(102, 302), y=sim_data_est['y']).set_title('Y')
plt.show()

# Plot X
plt.figure(figsize=(10, 6))
sns.lineplot(x=range(102, 302), y=sim_data_est['x']).set_title('X')
plt.show()

# Scatter plot of X and Y
plt.figure(figsize=(10, 6))
sns.scatterplot(x=sim_data_est['x'], y=sim_data_est['y']).set_title('Scatter Plot')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# OLS
X_est = sm.add_constant(sim_data_est['x'])
ols_fit = OLS(sim_data_est['y'], X_est).fit()
print(ols_fit.summary())

# Forecasts
X_fore = sm.add_constant(sim_data_fore['x'])
yhat_y = ols_fit.predict(X_fore)
yhat_se = np.sqrt(np.sum(ols_fit.resid**2) / (len(sim_data_est) - 2))
yhat_y_up = yhat_y + 1.96 * yhat_se
yhat_y_low = yhat_y - 1.96 * yhat_se

# Plot - actual (Y) & forecasts (YHAT)
plt.figure(figsize=(10, 6))
y_plot = pd.DataFrame({
    'x': list(range(302, 402)) * 2,
    'y': np.concatenate([sim_data_fore['y'], yhat_y]),
    'label': ['Y'] * 100 + ['YHAT'] * 100
})
sns.lineplot(data=y_plot, x='x', y='y', hue='label', style='label')
plt.show()

# Plot - yhat / yhat_up / yhat_low
plt.figure(figsize=(10, 6))
yhat_plot = pd.DataFrame({
    'x': list(range(302, 402)) * 3,
    'yhat': np.concatenate([yhat_y, yhat_y_up, yhat_y_low]),
    'label': ['YHAT'] * 100 + ['YHAT_UP'] * 100 + ['YHAT_LOW'] * 100
})
sns.lineplot(data=yhat_plot, x='x', y='yhat', hue='label', style='label')
plt.show()

# Recursive forecasts

# Initialize arrays for storing recursive forecasts and standard errors
yhat_y_rec = np.zeros(100)
yhat_y_recse = np.zeros(100)

# Recursive OLS forecasts
sim_data_rec = sm.add_constant(sim_data)
for i in range(100):
    # Prepare the training data
    train_data = sim_data_rec.iloc[101:(300 + i + 1)]
    X_rec = train_data[['const','x']]
    y_rec = train_data['y']
    
    # Fit the OLS model
    ols_rec = OLS(y_rec, X_rec).fit()
    
    # Prepare the new data for prediction
    X_new = sim_data_rec.iloc[[301 + i]][['const','x']]
    
    # Predict and calculate standard error
    yhat_y_rec[i] = ols_rec.predict(X_new).iloc[0]
    yhat_y_recse[i] = np.sqrt(np.sum(ols_rec.resid ** 2) / (197 + i))



# Combine actual and recursive forecasted values for plotting
y_values = np.concatenate((sim_data.iloc[301:401]['y'].values, yhat_y_rec))
labels = np.concatenate((['Y'] * 100, ['YHAT_REC'] * 100))
x_values = np.tile(np.arange(301, 401), 2)

# Create DataFrame for plotting
yrec_plot = pd.DataFrame({'x': x_values, 'y': y_values, 'label': labels})

# Plot - actual & recursive forecasts
plt.figure(figsize=(10, 6))
sns.lineplot(data=yrec_plot, x='x', y='y', hue='label', style='label')
plt.xlabel('')
plt.ylabel('')
plt.title('Actual & Recursive Forecasts')
plt.legend(title='')
plt.show()

# RMSE & MAE
yhat_Y = np.column_stack([yhat_y, yhat_y_rec])
RMSE = np.sqrt(np.mean((yhat_Y - sim_data_fore['y'].values[:, None])**2, axis=0))
MAE = np.mean(np.abs(yhat_Y - sim_data_fore['y'].values[:, None]), axis=0)
error_mat = pd.DataFrame([RMSE, MAE], index=['RMSE', 'MAE'], columns=['Simple', 'Recursive'])
print(error_mat)

### 1.12.1 Forecasting Euro Area GDP ###

eu_gdp = pd.read_csv('ex2_regress_gdp.csv')
eu_gdp['date'] = pd.to_datetime(eu_gdp['date'], format='%m/%d/%Y')

# Plots
plot_labels = ['96', '97', '98', '99', '00', '01', '02', '03', '04',
               '05', '06', '07', '08', '09', '10', '11', '12', '13']

fig, axes = plt.subplots(5, 1, figsize=(10, 20))
for i, var in enumerate(['y', 'ipr', 'sr', 'su', 'pr']):
    axes[i].plot(eu_gdp[var])
    axes[i].set_title(var.upper())
    axes[i].set_xticks([i*4 for i in range(18)])
    axes[i].set_xticklabels(plot_labels)
    plt.tight_layout()
    plt.show()

# Full sample - 1996Q1 to 2013Q2
gdp_formulas = [
    'y ~ ipr + su + pr + sr',
    'y ~ ipr + su + sr',
    'y ~ ipr + su',
    'y ~ ipr + pr + sr'
]
gdp_fits = [OLS.from_formula(formula, data=eu_gdp).fit() for formula in gdp_formulas]
for fit in gdp_fits:
    print(fit.summary())

# Estimation sample - 1996Q1 to 2006Q4
eu_gdp_est = eu_gdp.iloc[:44]
eu_gdp_fore = eu_gdp.iloc[44:]

gdp_ests = [OLS.from_formula(formula, data=eu_gdp_est).fit() for formula in gdp_formulas]
for est in gdp_ests:
    print(est.summary())

# Static and recursive forecasts
gdp_fore = [est.predict(eu_gdp_fore) for est in gdp_ests]

gdp_rec = []
for formula in gdp_formulas:
    rec_fore = np.zeros(26)
    for i in range(26):
        fit = OLS.from_formula(formula, data=eu_gdp.iloc[:(44 + i)]).fit()
        rec_fore[i] = fit.predict(eu_gdp.iloc[[44 + i]]).iloc[0]
    gdp_rec.append(rec_fore)

# Plots - actual & forecasts
plot_labels = [str(year) for year in range(2007, 2014)]
for model_idx in range(4):
    gdp_plot = pd.DataFrame({
        'Y': eu_gdp_fore['y'],
        f'YRFOREG{model_idx+1}': gdp_rec[model_idx],
        f'YFOREG{model_idx+1}': gdp_fore[model_idx]
    })

    fig, ax = plt.subplots()
    ax.plot(gdp_plot['Y'], label='Y', linestyle='-')
    ax.plot(gdp_plot[f'YRFOREG{model_idx+1}'], label=f'YRFOREG{model_idx+1}', linestyle='--')
    ax.plot(gdp_plot[f'YFOREG{model_idx+1}'], label=f'YFOREG{model_idx+1}', linestyle='-.')
    ax.legend()
    ax.set_xticklabels(plot_labels)
    plt.show()

# RMSE & MAE for GDP models
gdp_rec_Y = np.column_stack(gdp_rec)
RMSE = np.sqrt(np.mean((gdp_rec_Y - eu_gdp_fore['y'].values[:, None])**2, axis=0))
MAE = np.mean(np.abs(gdp_rec_Y - eu_gdp_fore['y'].values[:, None]), axis=0)
error_mat = pd.DataFrame([RMSE, MAE], index=['RMSE', 'MAE'], columns=[f'Model {i+1}' for i in range(4)])
print(error_mat)

### 1.12.2 Forecating US GDP ###

# Load the dataset
us_gdp = pd.read_csv('ex2_regress_gdp_us.csv')

# Convert the 'date' column to datetime format
us_gdp['date'] = pd.to_datetime(us_gdp['date'], format='%m/%d/%Y')

# Plot labels
plot_labels = ['85', '87', '89', '91', '93', '95', '97', '99', 
               '01', '03', '05', '07', '09', '11', '13']

# Variables to plot
variables = ['y', 'ipr', 'sr', 'su', 'pr']

# Create plots
for var in variables:
    plt.figure()
    plt.plot(us_gdp[var], linestyle='-')
    plt.title(var.upper())
    plt.xlabel('')
    plt.ylabel('')
    plt.xticks(ticks=[i * 8 for i in range(1, 16)], labels=plot_labels)
    plt.show()
    
# Full sample - 1985Q1 to 2013Q4
gdp_fit = []
gdp_formula = [
    'y ~ ipr + su + pr + sr', 
    'y ~ ipr + su + sr',
    'y ~ ipr + su', 
    'y ~ ipr + pr + sr'
]

# Summary statistics
k = [5, 4, 3, 4]
n = len(us_gdp)
summary_stat = pd.DataFrame({
    'R2': [0.0] * 4,
    'Adj R2': [0.0] * 4,
    'AIC': [0.0] * 4,
    'BIC': [0.0] * 4,
    'HQ': [0.0] * 4  
})
# Fit the models and print summaries
for model_index, formula in enumerate(gdp_formula):
    model = OLS.from_formula(formula, data=us_gdp).fit()
    gdp_fit.append(model)
    summary_stat.loc[model_index, 'R2'] = model.rsquared
    summary_stat.loc[model_index, 'Adj R2'] = model.rsquared_adj
    summary_stat.loc[model_index, 'AIC'] = model.aic + 2 # to make consistent with R
    summary_stat.loc[model_index, 'BIC'] = model.bic + 2 # to make consistent with R
    summary_stat.loc[model_index, 'HQ'] = model.aic + 2 * k[model_index] * (np.log(np.log(n)) - 1)
    print(model.summary())
    
# Transpose the DataFrame and set appropriate column names
summary_stat = summary_stat.T
summary_stat.columns = ['Model 1', 'Model 2', 'Model 3', 'Model 4']
print(summary_stat)


# Estimation sample - 1985Q1 to 2006Q4
us_gdp_est = us_gdp.iloc[:88]
us_gdp_fore = us_gdp.iloc[88:116]


gdp_est = []
for model in range(4):
    est_model = OLS.from_formula(gdp_formulas[model], data=us_gdp_est).fit()
    gdp_est.append(est_model)
    print(est_model.summary())
  
# Static and recursive forecasts
gdp_fore = []
gdp_rec = []

for model in range(4):
    # Static forecast
    fore_model = gdp_est[model].predict(us_gdp_fore)
    gdp_fore.append(fore_model)

    # Recursive forecast
    rec_forecasts = []
    for i in range(1, 29):
        # Refit model with increasing sample size
        rec_model = OLS.from_formula(gdp_formulas[model], data=us_gdp.iloc[:87 + i]).fit()
        rec_forecast = rec_model.predict(us_gdp.iloc[[87 + i]])
        rec_forecasts.append(rec_forecast.values[0])
    gdp_rec.append(rec_forecasts)
    

# Plot - actual & forecasts

# Define plot labels for x-axis
plot_labels = [str(year) for year in range(2007, 2014)]

# Static and recursive forecast plots for each model
for model_idx in range(4):
    # Combine data for plotting
    gdp_plot = pd.DataFrame({
        'Y': us_gdp_fore['y'],
        f'YRFOREG{model_idx+1}': gdp_rec[model_idx],
        f'YFOREG{model_idx+1}': gdp_fore[model_idx]
    })

    # Actual & static forecast plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(gdp_plot['Y'], label='Y', linestyle='-')
    ax.plot(gdp_plot[f'YFOREG{model_idx+1}'], label=f'YFOREG{model_idx+1}', linestyle='--')
    #ax.set_xticks(range(0, len(plot_labels) * 4, 4))
    ax.set_xticklabels(plot_labels)
    ax.set_ylim(-3, 2)
    ax.legend(loc='lower right', frameon=False)
    ax.set_title(f'Model {model_idx+1} - Actual & Static Forecast')
    plt.show()

    # Actual, recursive & static forecast plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(gdp_plot['Y'], label='Y', linestyle='-')
    ax.plot(gdp_plot[f'YRFOREG{model_idx+1}'], label=f'YRFOREG{model_idx+1}', linestyle='--')
    ax.plot(gdp_plot[f'YFOREG{model_idx+1}'], label=f'YFOREG{model_idx+1}', linestyle='-.')
    #ax.set_xticks(range(0, len(plot_labels) * 4, 4))
    ax.set_xticklabels(plot_labels)
    ax.set_ylim(-3, 2)
    ax.legend(loc='lower right', frameon=False)
    ax.set_title(f'Model {model_idx+1} - Actual, Recursive & Static Forecast')
    plt.show()
    
# Initialize lists for RMSE and MAE
RMSE = []
MAE = []

# Combine forecasts from all models
gdp_fore_Y = np.column_stack([gdp_fore[0], gdp_fore[1], gdp_fore[2], gdp_fore[3]])
gdp_rec_Y = np.column_stack([gdp_rec[0], gdp_rec[1], gdp_rec[2], gdp_rec[3]])

# Calculate RMSE and MAE for both static and recursive forecasts
RMSE_static = np.sqrt(np.sum((gdp_fore_Y - us_gdp_fore['y'].values[:, np.newaxis])**2, axis=0)) / np.sqrt(28)
RMSE_recursive = np.sqrt(np.sum((gdp_rec_Y - us_gdp_fore['y'].values[:, np.newaxis])**2, axis=0)) / np.sqrt(28)

MAE_static = np.sum(np.abs(gdp_fore_Y - us_gdp_fore['y'].values[:, np.newaxis]), axis=0) / 28
MAE_recursive = np.sum(np.abs(gdp_rec_Y - us_gdp_fore['y'].values[:, np.newaxis]), axis=0) / 28

# Combine RMSE and MAE into a matrix
error_mat = np.vstack([RMSE_static, MAE_static, RMSE_recursive, MAE_recursive])
error_df = pd.DataFrame(error_mat, index=['RMSE', 'MAE', 'RMSE recursive', 'MAE recursive'],
                        columns=['Model 1', 'Model 2', 'Model 3', 'Model 4'])

# Print the error matrix
print(error_df)

### 1.13.1 Forecasting default risk ###

# Load data
yield_data = pd.read_csv('ex3_regress_oas.csv')
yield_data['Date'] = pd.to_datetime(yield_data['Date'], format='%m/%d/%Y')

# Plot labels
plot_labels = ['98', '99', '00', '01', '02', '03', '04', '05', '06', 
               '07', '08', '09', '10', '11', '12', '13', '14', '15']

# Variables to plot
variables = ['OAS', 'VIX', 'SENT', 'sp500', 'PMI']

# Plot each variable
for var in variables:
    plt.figure()
    plt.plot(yield_data[var], linestyle='-', marker='None')
    plt.title(var)
    plt.xlabel('')
    plt.ylabel('')
    plt.xticks(ticks=range(0, len(yield_data), 12), labels=plot_labels, rotation=45, fontsize=8)
    plt.show()
    

# Create a copy of the original data to avoid modifying it directly
yield_data_copy = yield_data.copy()

# Number of months (rows) in the dataset
n_months = len(yield_data_copy)

# Lagged OAS data
oas_lagged = yield_data_copy.loc[1:n_months, 'OAS'].reset_index(drop=True)

# Adjust the original data to align with the lagged OAS data
yield_data_lagged = yield_data_copy.loc[:n_months-1].reset_index(drop=True)
yield_data_lagged['OAS'] = oas_lagged

# Display the updated dataframe with lagged OAS
#yield_data_lagged.head()

# Define the formulas for the models
yield_formulas = [
    'OAS ~ VIX',
    'OAS ~ SENT',
    'OAS ~ PMI',
    'OAS ~ sp500',
    'OAS ~ VIX + SENT + PMI + sp500'
]

# Fit the models and store the results
yield_fit = []
for formula in yield_formulas:
    model = OLS.from_formula(formula=formula, data=yield_data_lagged).fit()
    yield_fit.append(model)
    print(model.summary())
