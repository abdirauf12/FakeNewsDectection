#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# In[4]:


#Read the data
df = pd.read_csv('/Users/Abdi/Downloads/news.csv')

#Get shape of data
df.shape
df.head()


# In[6]:


labels = df.label
labels.head()


# In[9]:


#split data set into training and test
x_train,x_test,y_train,y_test= train_test_split(df['text'], labels, test_size= 0.2, random_state = 7)


# In[11]:


#initialize a TfidfVectorizer
tfidf_vectorizer= TfidfVectorizer(stop_words = 'english', max_df= 0.7)

#fit and transform a train set, transform a test set
tfidf_train = tfidf_vectorizer.fit_transform(x_train)
tfidf_test = tfidf_vectorizer.transform(x_test)


# In[12]:


#initialize a PassiveAgressiveClassifier
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)

#Predict on a test set and lculate the accuracy
y_pred = pac.predict(tfidf_test)
score = accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')


# In[13]:


confusion_matrix(y_test, y_pred, labels=['FAKE','REAL'])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




