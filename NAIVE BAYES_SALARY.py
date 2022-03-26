#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
import warnings
warnings.filterwarnings("ignore")


# In[2]:


Test = pd.read_csv('SalaryData_Test.csv')

Train = pd.read_csv('SalaryData_Train.csv')


# In[3]:


Train.head()


# In[4]:


Test.head()


# In[5]:


from sklearn.preprocessing import LabelEncoder
Train = Train.apply(LabelEncoder().fit_transform)
Train.head()


# In[6]:


from sklearn.preprocessing import LabelEncoder
Test = Test.apply(LabelEncoder().fit_transform)
Test.head()


# In[7]:


Train.info()


# In[8]:


Test.info()


# In[9]:


Train.shape 


# In[10]:


Test.shape


# In[11]:


Train.columns


# In[ ]:





# In[12]:


Test[Test.duplicated()]


# In[13]:


Test2 = Test.drop_duplicates()
Test2.shape


# In[14]:


Train[Train.duplicated()]


# In[15]:


Train2 = Train.drop_duplicates()
Train2.shape


# In[ ]:





# In[ ]:





# In[16]:


#To find out the correlation between all features
import seaborn as sns
plt.figure(figsize= (16, 7))
sns.heatmap(Train.corr(), annot= True);


# In[17]:


import seaborn as sns
plt.figure(figsize= (16, 7))
sns.heatmap(Test.corr(), annot= True);


# In[18]:


sns.countplot(x= 'Salary', data= Train )


# In[19]:


sns.countplot(x= 'Salary', data= Test )


# In[ ]:





# In[ ]:





# In[20]:


X_train= Train2.drop(['education','relationship','native','maritalstatus','sex','race'],axis=1).values
Y_train= Train2['Salary'].values
print(np.unique(Y_train))
X_train


# In[21]:


X_test= Test2.drop(['education','relationship','native','maritalstatus','sex','race'],axis=1).values
Y_test= Test2['Salary'].values
print(np.unique(Y_test))
X_test


# In[22]:


from sklearn.naive_bayes import GaussianNB 
gnb = GaussianNB() 
gnb.fit(X_train, Y_train) 
y_pred = gnb.predict(X_test) 
 
from sklearn import metrics 
print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(Y_test, y_pred)*100)


# In[23]:


from sklearn.naive_bayes import MultinomialNB as MB
from sklearn.naive_bayes import GaussianNB as GB


# In[24]:


classifier_mb = MB()
classifier_mb.fit(X_train, Y_train)
classifier_mb.score(X_train, Y_train)
classifier_mb.score(X_test, Y_test)
predicted_result = classifier_mb.predict(X_train)
accuracy_train = np.mean(predicted_result == Y_train)
accuracy_train


# In[25]:


test_predict=classifier_mb.predict(X_test)
accuracy_test = np.mean(test_predict== Y_test)
accuracy_test


# In[26]:


# Gaussian Naive Bayes
classifier_gb = GB()
classifier_gb.fit(X_train, Y_train)
classifier_gb.score(X_train, Y_train)
classifier_gb.score(X_test, Y_test)
train_pred = classifier_gb.predict(X_train)
accuracy_train = np.mean(train_pred == Y_train)
accuracy_train


# In[27]:


test_pred=classifier_gb.predict(X_test)
accuracy_test = np.mean(test_pred== Y_test)
accuracy_test


# In[28]:


print("Accuracy:",metrics.accuracy_score(Y_test, y_pred))


# In[29]:


print("Training_set _score: {:.2f}".format(classifier_gb.score(X_train, Y_train)))


# In[30]:


print("Training_set _score: {:.2f}".format(classifier_gb.score(X_test, Y_test)))

