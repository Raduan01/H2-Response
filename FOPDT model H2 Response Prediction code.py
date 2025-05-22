#!/usr/bin/env python
# coding: utf-8

# # Import necessary libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import time


# # Load the CSV file

# In[ ]:


file_path = ''  # Replace with CSV data
data = pd.read_csv(file_path)


# # Eliminate dead time (non-zero H2 Response)

# In[ ]:


non_zero_index = data[data['H2'] >0].index[0]


# # Use a small time window vlaues

# In[ ]:


data_after_dead_time = data.iloc[non_zero_index:].copy() 
data_few_points = data_after_dead_time.head(535).copy() #use small time window to predict the response

data_few_points['time_since_dead_time'] = np.arange(len(data_few_points))
time_values = data_few_points['time_since_dead_time'].values
H2_values = data_few_points['H2'].values


# # FOPDT model predictions

# In[ ]:


def fopdt_model(t, K, tau, theta):
    return K * (1 - np.exp(-(t - theta) / tau)) * (t >= theta) #K is the predicted Final resposne of the sensor

initial_guess = [1.0, 10.0, 0] 
params, covariance = curve_fit(fopdt_model, time_values, H2_values, p0=initial_guess)
K_estimated, tau_estimated, theta_estimated = params
time_values_full = np.arange(0, len(data_after_dead_time))  
H2_predicted_full = fopdt_model(time_values_full, K_estimated, tau_estimated, theta_estimated)
data_after_dead_time['time_since_dead_time'] = np.arange(len(data_after_dead_time))
time_values_after_few_points = np.arange(len(data_few_points), 6500)  # Time after the few points up to 7200
H2_predicted_after_few_points = fopdt_model(time_values_after_few_points, K_estimated, tau_estimated, theta_estimated)
data_after_dead_time['time_since_dead_time'] = np.arange(len(data_after_dead_time))


# # Plotting all the values

# In[ ]:


plt.figure(figsize=(12, 8))

plt.plot(data_after_dead_time['time_since_dead_time'], data_after_dead_time['H2'], color='green', linewidth=1, label='Sensor Response')
plt.plot(data_few_points['time_since_dead_time'], H2_values, color='blue',linewidth=6, label='Time Window Value')
plt.plot(time_values_after_few_points, H2_predicted_after_few_points, color='red',linewidth=5,  label=f"FOER prediction" )



plt.axvline(x=1138, color='m',label=r"${\mathrm{t}_{90}}$ time")
plt.axvline(x=534, color='y',linestyle='--',label=r"Time Window")
plt.axhline(y=0, color='black',linestyle='--',label=r"Time Window Vol % Start")
plt.axhline(y=0.5, color='cyan',linestyle='--',label=r"Time Window Vol % End")
plt.xticks(fontsize=25)
plt.yticks(fontsize=20)

plt.title(f"Prediction by FO with Early Response Model",fontsize=25)
plt.xlabel("Time t in [s]",fontsize=25)
plt.ylabel("Volume Fraction in [Vol%]",fontsize=25)
plt.legend(fontsize=22,bbox_to_anchor=(0.2, 0.7),)
plt.show()


# # Display the estimated K, tau, and dead time theta

# In[ ]:


print(f"Estimated K: {K_estimated:.2f}") #sensor stable prediction vlaues
print(f"Estimated Tau: {tau_estimated:.2f}") # Time constant prediction vlaues
print(f"Estimated Dead Time (Theta): {theta_estimated:.2f}")


# In[ ]:




