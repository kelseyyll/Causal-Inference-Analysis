# Simulate Time Series Data to mimick weekly store sales from chain retail store for Causal Inference Project
# The sales values similuted here are for illustration purpose, and they could be greatly different from real retail store sales
# All the time series data have a length of 104 weeks (2 years)
# T1 to T4 as reference store sales time series
# The period of last 26 weeks was treated as event period
# T_target is the time series that needs to be evaluted for event effet, assuming implemented marketing strategy improvement in the last 26 weeks for Store T only
# Seasonal spikes are created for all the time series

########################################################################################################################################
# Create a model to predict whether the marketing strategy improvement made any difference in sales using Python CausalImpact package  #
########################################################################################################################################

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


##################################################################
#                      Data Simulation                           #
##################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(0)

# Define the number of weeks
num_weeks = 104

# Create a time index for 104 weeks starting from January 1, 2022
time_index = pd.date_range(start='2022-01-01', periods=num_weeks, freq='7D')

# Create a week index variable ranging from 1 to 104
week_index = np.arange(1, num_weeks + 1)

# Function to simulate time series with spikes for holidays
def simulate_time_series():
    t = np.random.normal(0, 1, num_weeks)  # Random noise
    holidays = pd.to_datetime(['2022-07-04', '2022-09-05', '2022-11-24', '2022-12-25',
                               '2023-07-04', '2023-09-04', '2023-11-24', '2023-12-25' ])  # July 4th, Labor Day, Thanksgiving, and Christmas
    holiday_idx = time_index.isin(holidays)  # Identify holidays
    t[holiday_idx] += np.random.normal(3, 0.1, num_weeks)[holiday_idx]  # Large positive spikes on holidays
    return t

# Define constants for T2, T3, T4, and T_target for set some base weekly sales
constant = 5000

t0 = simulate_time_series()*1000

# Simulate T1
t1 = t0 + constant

# Simulate T2 as t0 + 1.1*constant + random noises
t2 = t0 + 1.1*constant + np.random.normal(0, 1000, num_weeks)

# Simulate T3 as a series with new seed + 1.2 constant + random noises
np.random.seed(101)
t3 = simulate_time_series()*1000 + 1.2 * constant + np.random.normal(0, 700, num_weeks)

# Simulate T4 with a new seed + 1.3 * constant + larger random noises
np.random.seed(42)
t4 = simulate_time_series()*1000 + 1.3 * constant + np.random.normal(0, 1700, num_weeks)

# Simulate T_target as T1 + 0.15*constant + smallerrandom noises
t_target = t1 + 0.15*constant + np.random.normal(200, 500, num_weeks)

# Add artifical lift to the last 26 weeks to the target serires (to mimick the event effect)
weekly_avg = t_target[:-26].mean()
t_target[-26:] += 0.35 * weekly_avg

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


####################################################################################
#                              Time Series EDA                                     #
####################################################################################

df = data
df.head()

# to find the best Reference Time Series (Control Store weekly sales) that can best mimick
# the pattern of the target store sales before event (the marketing strategy improvement period)

# calculate the correlation matrix (before event)
df1 = df[(df['Flag']==0)][['T_target','T1', 'T2', 'T3','T4']]

df1.head()

corr_matrix = df1.corr()

# plot the heatmap
sns.set(font_scale=0.8)
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')

# display the plot
plt.show()

# T1 has the highest correlation with the target store sales. 
# Plot T_target store sales (response series) along with the time series that has the highest correlations 

# to set the plot size 
plt.figure(figsize=(16, 8), dpi=150) 
  
# set the label and color of the curve. 
df1['T_target'].plot(label='Response Series', color='Red') 
df1['T1'].plot(label='Reference Store 1', color='Blue') 
  
# adding title to the plot 
plt.title('Pre-Period Weekly Sales: Target Store with Reference Store1') 
  
# adding Label to the x-axis 
plt.xlabel('Week') 
  
# adding legend to the curve 
plt.legend() 


##############################################################################################
#             Fit the Causal Inference Model with T1 as Reference/Control Series             #
##############################################################################################

# df.set_index('Week Index')

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

# print(CI_T1.summary('report')) ## might not work depending on the version of the package
# CI_T1.inferences

# The model predicted significant weekly sales increase for the event period


###############################################################################################
#    Explore the method with a counterexample (serves as alternative for model evaluation)    #
###############################################################################################

# It would be interesting to see how the method can generate if we trick the algoritm to use the weekly store sales from a store 
# where the local promotion improvement was not implemented as the Target Time Series. We can use T1 and T2, protending T1 is the target series. 

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

# The model predicted that the incremental sales for event period was close to zero.
# The store without marketing strategy improvement failed to generate any incremental sales. 
# The 95% Confidence Interval for Absolute Effect includes 0.
# The model did find out that the "fake" target series (store T) is a "fake"


