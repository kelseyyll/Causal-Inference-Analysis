# Causal-Inference-Analysis

## Project Overview
This project applies structural Bayesian time series analysis to evaluate the impact of marketing strategies on grocery store sales. It's an independent research project aimed at understanding the effects of marketing changes using simulated data.

## Description
The project simulates time series data, representing weekly sales of a chain grocery store. The key focus is on:
- Simulating reference store sales time series (T1 to T4) and a target time series (T_target) for event effect evaluation.
- Analyzing the impact of a marketing strategy introduced in the last 26 weeks of a 104-week period.
- Employing the Python CausalImpact package to estimate causal effects in a non-experimental setting.

## Key Features
- **Data Simulation**: Involves generating time series data with seasonal spikes and fluctuations.
- **Causal Impact Analysis**: Uses CausalImpact to assess the marketing strategy's effect on sales.
- **Visualization**: The project includes visualizations to explore time series trends, seasonality, and correlation.

## Usage
Run the provided `.py` script to simulate the data, perform the analysis, and visualize the results. Ensure you have the necessary Python packages installed, including pandas, numpy, matplotlib, seaborn, and CausalImpact.

## Author
- Kelsey Lin

## Contact
For any inquiries regarding this project, feel free to reach out at kelseylin078@gmail.com
