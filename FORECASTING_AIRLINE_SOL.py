#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing # SES
from statsmodels.tsa.holtwinters import Holt # Holts Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing # 
import statsmodels.graphics.tsaplots as tsa_plots
import statsmodels.tsa.statespace as tm_models
from datetime import datetime,time
import warnings
warnings.filterwarnings("ignore")


# In[2]:


Air = pd.read_excel("Airlines+Data.xlsx")


# In[3]:


Air.head()


# In[4]:


Air.tail()


# In[5]:


Air.info()


# In[6]:


Air.value_counts()


# In[7]:


Air.shape


# In[8]:


# Converting the normal index of Amtrak to time stamp 
Air.index = pd.to_datetime(Air.Month,format="%b-%y")


# In[9]:


Air.Passengers.plot()


# In[10]:


# Creating a Date column to store the actual Date format for the given Month column
Air["Date"] = pd.to_datetime(Air.Month,format="%b-%y")


# In[11]:


# Extracting Day, weekday name, month name, year from the Date column using 
# Date functions from pandas 

Air["month"] = Air.Date.dt.strftime("%b") # month extraction

#Air["Day"] = Air.Date.dt.strftime("%d") # Day extraction

#Air["wkday"] = Air.Date.dt.strftime("%A") # weekday extraction

Air["year"] = Air.Date.dt.strftime("%Y") # year extraction


# In[12]:


# Some EDA on Time series data 
# Heat map visualization 
heatmap_y_month = pd.pivot_table(data=Air,values="Passengers",index="year",columns="month",aggfunc="mean",fill_value=0)
sns.heatmap(heatmap_y_month,annot=True,fmt="g")


# In[13]:


# Boxplot for ever
sns.boxplot(x="month",y="Passengers",data=Air)
sns.boxplot(x="year",y="Passengers",data=Air)


# In[14]:


# Line plot for Ridership based on year  and for each month
sns.lineplot(x="year",y="Passengers",hue="month",data=Air)


# In[15]:


# moving average for the time series to understand better about the trend character in Airline
Air.Passengers.plot(label="org")
for i in range(2,24,6):
    Air["Passengers"].rolling(i).mean().plot(label=str(i))
plt.legend(loc=3)


# In[16]:


# Time series decomposition plot 
decompose_ts_add = seasonal_decompose(Air.Passengers,model="additive")
decompose_ts_add.plot()
decompose_ts_mul = seasonal_decompose(Air.Passengers,model="multiplicative")
decompose_ts_mul.plot()


# In[17]:


# ACF plots and PACF plots on Original data sets 
tsa_plots.plot_acf(Air.Passengers,lags=10)
tsa_plots.plot_pacf(Air.Passengers)


# In[18]:


# Air.index.freq = "MS" 
# splitting the data into Train and Test data and considering the last 12 months data as 
# Test data and left over data as train data 

Train = Air.head(133)
Test = Air.tail(12)


# In[19]:


# Creating a function to calculate the MAPE value for test data 
def MAPE(pred,org):
    temp = np.abs((pred-org))*100/org
    return np.mean(temp)


# In[20]:


# Simple Exponential Method
ses_model = SimpleExpSmoothing(Train["Passengers"]).fit()
pred_ses = ses_model.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_ses,Test.Passengers)


# In[21]:


# Holt method 
hw_model = Holt(Train["Passengers"]).fit()
pred_hw = hw_model.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hw,Test.Passengers)


# In[22]:


# Holts winter exponential smoothing with additive seasonality and additive trend
hwe_model_add_add = ExponentialSmoothing(Train["Passengers"],seasonal="add",trend="add",seasonal_periods=12,damped=True).fit()
pred_hwe_add_add = hwe_model_add_add.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hwe_add_add,Test.Passengers)


# In[23]:


# Holts winter exponential smoothing with multiplicative seasonality and additive trend
hwe_model_mul_add = ExponentialSmoothing(Train["Passengers"],seasonal="mul",trend="add",seasonal_periods=12).fit()
pred_hwe_mul_add = hwe_model_mul_add.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hwe_mul_add,Test.Passengers)


# In[25]:


pip install pmdarima


# In[26]:


# Lets us use auto_arima from p
from pmdarima import auto_arima
auto_arima_model = auto_arima(Train["Passengers"],start_p=0,
                              start_q=0,max_p=10,max_q=10,
                              m=12,start_P=0,seasonal=True,
                              d=1,D=1,trace=True,error_action="ignore",
                              suppress_warnings= True,
                              stepwise=False)


# In[27]:


auto_arima_model.summary()


# In[28]:


# For getting Fitted values for train data set we use 
# predict_in_sample() function 
auto_arima_model.predict_in_sample( )


# In[29]:


# For getting predictions for future we use predict() function 
pred_test = pd.Series(auto_arima_model.predict(n_periods=12))
# Adding the index values of Test Data set to predictions of Auto Arima
pred_test.index = Test.index
MAPE(pred_test,Test.Passengers)


# In[31]:


# Visualization of Forecasted values for Test data set using different methods 
plt.plot(Train.index, Train["Passengers"], label='Train',color="black")
plt.plot(Test.index, Test["Passengers"], label='Test',color="blue")
plt.plot(pred_ses.index, pred_ses, label='SimpleExponential',color="green")
plt.plot(pred_hw.index, pred_hw, label='Holts_winter',color="red")
plt.plot(pred_hwe_add_add.index,pred_hwe_add_add,label="HoltsWinterExponential_1",color="brown")
plt.plot(pred_hwe_mul_add.index,pred_hwe_mul_add,label="HoltsWinterExponential_2",color="yellow")
plt.plot(pred_hwe_mul_add.index,pred_hwe_mul_add,label="Auto_Arima",color="grey")
plt.legend(loc='best')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




