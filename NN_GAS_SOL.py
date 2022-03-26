#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Flatten, LeakyReLU, BatchNormalization
from keras.wrappers.scikit_learn import KerasClassifier
from keras.activations import relu, sigmoid


# In[2]:


GT = pd.read_csv("gas_turbines.csv")


# In[3]:


GT.head()


# In[5]:


GT.isnull().sum()


# In[11]:


GT.shape


# In[12]:


GT.info()


# In[13]:


GT.columns


# In[24]:


GT.describe()


# In[25]:


#To find out the correlation between all features
import seaborn as sns
plt.figure(figsize= (16, 7))
sns.heatmap(GT.corr(), annot= True);


# In[27]:


from sklearn.preprocessing import MinMaxScaler
scale=MinMaxScaler()
gt=pd.DataFrame(scale.fit_transform(GT),columns=['AT','AP','AH','AFDP','GTEP','TIT','TAT','TEY','CDP','CO','NOX'])
gt


# In[28]:


X = gt.drop(axis=0,columns="TEY").values
Y = gt["TEY"].values


# In[29]:


seed = 7
np.random.seed(seed)
model = Sequential()
model.add(Dense(12, input_dim=10, kernel_initializer='uniform', activation='relu'))
model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))


# In[30]:


model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])


# In[31]:


model.fit(X, Y, validation_split=0.33, epochs=100, batch_size=10)


# In[32]:


scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# In[33]:


history = model.fit(X, Y, validation_split=0.33, epochs=100, batch_size=10)


# In[34]:


model.history.history.keys()


# In[35]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[36]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[37]:


from sklearn.model_selection import GridSearchCV, KFold
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import Adam


# In[40]:


def create_model():
    model = Sequential()
    model.add(Dense(12, input_dim=10, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
    
    adam=Adam(learning_rate=0.01)
    model.compile(loss='mse', optimizer=adam, metrics=['accuracy'])
    return model


# In[47]:


model = KerasClassifier(build_fn = create_model,verbose = 0)
batch_size = [10,20,40]
epochs = [10,50,100]
param_grid = dict(batch_size = batch_size,epochs = epochs)
grid = GridSearchCV(estimator = model,param_grid = param_grid,cv = KFold(),verbose = 10)
grid_result = grid.fit(X,Y)


# In[63]:


def create_model():
    model = Sequential()
    model.add(Dense(12, input_dim=10, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
    
    adam=Adam(beta_1=0.9)
    model.compile(loss='mse', optimizer=adam, metrics=['accuracy'])
    return model


# In[64]:


model = KerasClassifier(build_fn = create_model,verbose = 0)
batch_size = [10,20,40]
epochs = [10,50,100]
param_grid = dict(batch_size = batch_size,epochs = epochs)
grid = GridSearchCV(estimator = model,param_grid = param_grid,cv = KFold(),verbose = 10)
grid_result = grid.fit(X,Y)


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





# In[ ]:




