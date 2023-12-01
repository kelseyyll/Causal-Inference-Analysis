# Simulate Time Series Data to mimick weekly store sales from chain retail store for Causal Inference Project

# T1 to T4 as reference store sales time series
# T_target is the time series that needs to be evaluted for event effect
# The last 26 weeks was treated as event period
# T_Target is referring to the store that implemented marketing strategy improvement in the last 26 weeks
# Seasonal spikes are created for all the time series
# All the time series data have a length of 104 weeks (2 years)

# Create a model to predict whether the marketing strategy improvement made any difference in sales using Python CausalImpact package

# The Python CausalImpact package implements an approach to estimating the causal effect of a designed intervention on a time series. 
# Without randomized experiment, estimating the effect of a interventioin is very challenging. 
# The package aims to address this difficulty using a structural Bayesian time-series model 
# to estimate how the response metric might have evolved after the intervention if the intervention had not occurred.

pip install --upgrade pip
pip install CausalImpact
pip install tensorflow

from causalimpact import CausalImpact
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from statsmodels.tsa.seasonal import seasonal_decompose

from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

%matplotlib inline
sns.set_context('poster')
sns.set_style('darkgrid')
warnings.filterwarnings('ignore')





# Data Simulation

# Set random seed for reproducibility
np.random.seed(0)

# Define the number of weeks
num_weeks = 104

# Create a time index for 104 weeks starting from January 1, 2022
time_index = pd.date_range(start='2022-01-01', periods=num_weeks, freq='7D')

# Create a week index variable ranging from 1 to 104
week_index = np.arange(1, num_weeks + 1)

# Function to simulate time series with spikes for weekends and holidays
def simulate_time_series():
    t = np.random.normal(0, 1, num_weeks)  # Random noise
    weekends = np.array([i % 7 >= 5 for i in range(num_weeks)])  # Identify weekends
    holidays = pd.to_datetime(['2022-07-04', '2022-09-05', '2022-11-24', '2022-12-25',
                               '2023-07-04', '2023-09-04', '2023-11-24', '2023-12-25' ])  # July 4th, Labor Day, Thanksgiving, and Christmas
    holiday_idx = time_index.isin(holidays)  # Identify holidays
    t[weekends] += np.random.normal(0, 0.5, num_weeks)[weekends]  # Small spikes on weekends
    t[holiday_idx] += np.random.normal(0, 2, num_weeks)[holiday_idx]  # Large spikes on holidays
    return t

# Define constants for T2, T3, T4, and T_target
constant = 2500

# Simulate T1
t1 = simulate_time_series()*2000

# Simulate T2 as T1 + constant + random noises
t2 = t1 + constant + np.random.normal(0, 2000, num_weeks)

# Simulate T3 as a series with new seed + 1.2 constant + random noises
np.random.seed(101)
t3 = simulate_time_series()*2000 + 1.2* constant + np.random.normal(0, 1700, num_weeks)

# Simulate T4 with a new seed + 1.5 constant + random noises
np.random.seed(42)
t4 = simulate_time_series()*2000 + 1.5 * constant + np.random.normal(0, 2700, num_weeks)

# Simulate T_target as T1 + 1.7 * constant + random noises
t_target = t1 + 1.7 * constant + np.random.normal(0, 1500, num_weeks)

# Add artifical lift to the last 26 weeks to the target serires (mimick the event effect)
weekly_avg = t_target[:-26].mean()
t_target[-26:] += 0.6 * weekly_avg

# Create a flag to identify the last 26 weeks (1 for the last 26 weeks, 0 for the rest)
flag = [1 if i >= (num_weeks - 26) else 0 for i in range(num_weeks)]

# Create a DataFrame for the week index, time index, and the time series data
data = pd.DataFrame({'Week Index': week_index, 'Time Index': time_index, 'T1': t1, 'T2': t2, 'T3': t3, 'T4': t4, 'T_target': t_target, 'Flag': flag})

# Plot the time series
plt.figure(figsize=(14, 6))
for column in data.columns[2:-1]:  # Exclude the 'Week Index', 'Time Index', and 'Flag' columns
    plt.plot(data['Week Index'], data[column], label=column)
plt.title('Time Series Data')
plt.xlabel('Week Index')
plt.ylabel('Value')
plt.grid(True)
plt.legend()
plt.show()

# Display the first few rows of the DataFrame
print(data.head())





# Time Series EDA

df = data
df.head()
# df.tail()

# to find the best Reference Time Series (Control Store weekly sales) that can best mimick
# the pattern of the target store sales before the marketing strategy improvement

# calculate the correlation matrix
df1 = df[(df['Flag']==0)][['T_target','T1', 'T2', 'T3','T4']]

df1.head()

corr_matrix = df1.corr()

# plot the heatmap
sns.set(font_scale=0.8)
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')

# display the plot
plt.show()

# T1 and T2 have the highest correlation with the target store sales. 
# Plot T_target store sales (response series) along with the time series that has the highest correlations 

# Plot the response series along with the time series that has the highest correlation 
  
# to set the plot size 
plt.figure(figsize=(16, 8), dpi=150) 
  
# in plot method we set the label and color of the curve. 
df1['T_target'].plot(label='Response Series', color='Red') 
df1['T1'].plot(label='Reference Store 1', color='Blue') 
  
# adding title to the plot 
plt.title('Pre-Period Weekly Sales: Target Store with Reference Store1') 
  
# adding Label to the x-axis 
plt.xlabel('Week') 
  
# adding legend to the curve 
plt.legend() 


# Fit the Causal Inference Model with T1 as Reference/Control Series
df.set_index('Week Index')

# Fit the Causal Inference Model using Causal Impact
pre_period = [1, 78] 

# Define post-event period (time AFTER the event occurred)
post_period = [79, 104]


# Reference (Control): df['T1']
CI_T1 = CausalImpact(data = df[['T_target', 'T1']], 
                  pre_period = pre_period, 
                  post_period = post_period,
                  model_args={"ndraw":10000,
                             "prior_level_sd":None},
                  alpha = 0.1)


CI_T1.run() 
CI_T1.summary()
CI_T1.plot()
# print(CI_T1.summary('report')) ## this might not work depending on the version of the package

# CI_T1.inferences


# Explore the method with Counter Example (serves as alternative for model evaluation)
# It would be interesting to see how the method can generate if we trick the algoritm to use the weekly store sales from a store 
# where the local promotion improvement was not implemented as the Target Time Series. We can use T1 and T2, protending T1 is the response. 

## Fit the Causal Inference Model using Causal Impact
pre_period = [1, 78] 

# Define post-event period - i.e. time AFTER the event occurred.
post_period = [79, 104]

CI_CounterExp = CausalImpact(data = df[['T1', 'T2']], 
                  pre_period = pre_period, 
                  post_period = post_period,
                  model_args={"ndraw":10000,
                             "prior_level_sd":None},  
                  alpha = 0.1)


CI_CounterExp.run()
CI_CounterExp.summary()
CI_CounterExp.plot()

# The store without marketing strategy improvement failed to see any incremental sales. 
# The 95% Confidence Interval for Absolute Effect includes 0.
# The model did find out that the "fake" target store is a "fake"


