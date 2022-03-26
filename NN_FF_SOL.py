#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Flatten, LeakyReLU, BatchNormalization, 
from keras.wrappers.scikit_learn import KerasClassifier
from keras.activations import relu, sigmoid


# In[ ]:





# In[2]:


ff= pd.read_csv('forestfires2.csv')


# In[3]:


ff.head()


# In[4]:


ff.info()


# In[5]:


ff.columns


# In[6]:


ff[ff.duplicated()]


# In[7]:


ff[ff.duplicated()].shape


# In[8]:


ff = ff.drop_duplicates()


# In[9]:


ff.head()


# In[10]:


F = ff.drop(labels= ['dayfri', 'daymon', 'daysat', 'daysun', 'daythu',
       'daytue', 'daywed', 'monthapr', 'monthaug', 'monthdec', 'monthfeb',
       'monthjan', 'monthjul', 'monthjun', 'monthmar', 'monthmay', 'monthnov',
       'monthoct', 'monthsep'], axis=1)


# In[11]:


categorical= [i for i in ff.columns if ff[i].dtypes=='O']  #to sperate categorical features from data and creating a loop in which types should be "O" i.e. Objects


# In[12]:


categorical


# In[13]:


for i in categorical:
    print(f"{i} : {len(ff[i].unique())}")


# In[14]:


for feature in categorical:
    print(ff[feature].value_counts())
    print("\n \n")


# In[15]:


#Changing size_category into numerical data to find out the correlation
F['size_category']= F['size_category'].replace({'small':0, 'large':1}, regex=True)


# In[16]:


F


# In[17]:


#To find out the correlation between all features
import seaborn as sns
plt.figure(figsize= (16, 7))
sns.heatmap(F.corr(), annot= True);


# In[18]:


#Since there is not much correlation among features we cannot drop any column
#Therefore we will apply label encoder

from sklearn.preprocessing import LabelEncoder
F = F.apply(LabelEncoder().fit_transform)
F


# In[19]:


sns.countplot(x= 'size_category', data= F )


# In[30]:


x = F.iloc[:,0:10].values
y = F['size_category'].values


# In[31]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y)


# In[32]:


#The transformations to the data:
from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
X_train= sc.fit_transform(X_train)
X_test= sc.transform(X_test)


# In[33]:


from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=100,)

mlp.fit(X_train,y_train)
prediction_train=mlp.predict(X_train)
prediction_test = mlp.predict(X_test)


# In[34]:


from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,prediction_test))
np.mean(y_test==prediction_test)
np.mean(y_train==prediction_train)


# In[35]:


seed = 7
np.random.seed(seed)


# In[36]:


model = Sequential()
model.add(Dense(12, input_dim=10, kernel_initializer='uniform', activation='relu'))
model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='linear'))


# In[37]:


model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])


# In[38]:


model.fit(x, y, validation_split=0.33, epochs=100, batch_size=10)


# In[39]:


scores = model.evaluate(x, y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# In[50]:


history = model.fit(x, y, validation_split=0.33, epochs=100, batch_size=10)


# In[51]:


model.history.history.keys()


# In[52]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[67]:


def create_model():
    model = Sequential()
    model.add(Dense(12, input_dim=10, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(1, kernel_initializer='uniform', activation='linear'))
    
    adam=Adam(learning_rate=0.01)
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    return model


# In[68]:


from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import Adam


# In[69]:


from sklearn.model_selection import KFold
model = KerasClassifier(build_fn = create_model,verbose = 0)
batch_size = [10,20,40]
epochs = [10,50,100]
param_grid = dict(batch_size = batch_size,epochs = epochs)
grid = GridSearchCV(estimator = model,param_grid = param_grid,cv = KFold(),verbose = 10)
grid_result = grid.fit(x,y)


# In[ ]:




