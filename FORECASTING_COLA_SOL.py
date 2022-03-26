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


C = pd.read_excel("CocaCola_Sales_Rawdata.xlsx")


# In[3]:


C.head()


# In[4]:


C.info()


# In[5]:


C.isnull().sum()


# In[6]:


C[C.duplicated()].sum()


# In[7]:


C.shape


# In[8]:


C.columns


# In[9]:


C.describe()


# In[10]:


C_visual = pd.read_excel('CocaCola_Sales_Rawdata.xlsx',header=0,index_col=0,parse_dates=None,squeeze=True)


# In[11]:


C_visual.plot()


# In[12]:


C['Quarters']= 0
C['Year'] = 0
for i in range(42):
    p = C["Quarter"][i]
    C['Quarters'][i]= p[0:2]
    C['Year'][i]= p[3:5]
C.head()


# In[28]:


Quarter_dummies = pd.DataFrame(pd.get_dummies(C['Quarters']))
Quarter_dummies.head()


# In[29]:



c = pd.concat([C,Quarter_dummies],axis = 1)

c["t"] = np.arange(1,43)

c["t_squared"] = c["t"]*c["t"]
c.columns
c["log_sales"] = np.log(c["Sales"])
c.Sales.plot()
Train = c.head(35)
Test = c.tail(7)


# In[14]:


plt.figure(figsize=(12,10))
plot_quarter_y = pd.pivot_table(data =C,values="Sales",index="Year",columns="Quarters",aggfunc="mean",fill_value=0)
sns.heatmap(plot_quarter_y,annot=True,fmt = "g")


# In[15]:


sns.boxplot(x='Year',y='Sales',data=C)


# In[16]:


sns.lineplot(x='Year',y='Sales',data=C)


# In[17]:


import statsmodels.formula.api as smf 

#LINEAR
linear_model = smf.ols('Sales~t',data=Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_linear))**2))
rmse_linear


# In[18]:


#Exponential
Exp = smf.ols('log_sales~t',data=Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp


# In[19]:


#Quadratic

Quad = smf.ols('Sales~t+t_squared',data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_squared"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_Quad))**2))
rmse_Quad


# In[32]:


#Additive seasonality
add_sea_model = smf.ols('Sales~Q1+Q2+Q3+Q4',data=Train).fit()
pred_add_sea = pd.Series(add_sea_model.predict(Test[['Q1','Q2','Q3','Q4']]))
pred_add_sea
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea))**2))
rmse_add_sea


# In[35]:


#Additive Seasonality Quadratic
add_sea_Quad = smf.ols('Sales~t+t_squared+Q1+Q2+Q3+Q4',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['Q1','Q2','Q3','Q4','t','t_squared']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad 


# In[37]:


#Multiplicative Seasonality
Mul_sea = smf.ols('log_sales~+Q1+Q2+Q3+Q4',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea


# In[40]:


#Multiplicative Additive Seasonality
Mul_Add_sea = smf.ols('log_sales~t+Q1+Q2+Q3+Q4',data = Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea 


# In[41]:


#Testing 

data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
table_rmse=pd.DataFrame(data)
table_rmse

