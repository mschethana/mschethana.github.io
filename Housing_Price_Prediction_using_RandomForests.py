#!/usr/bin/env python
# coding: utf-8

# In[166]:


import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.metrics import mean_absolute_error,r2_score
plt.style.use('ggplot')


# In[167]:


#to read the kaggle dataset "Housing.csv"
housing_data=pd.read_csv('Housing.csv')


# In[168]:


housing_data.head()


# In[169]:


housing_data.shape


# In[170]:


housing_data.columns


# In[171]:


#from above output, we can see that there are 13 features and 545 instances (that is, rows and columns)


# In[172]:


housing_data.dtypes


# In[173]:


housing_data.info()


# In[174]:


#from above output, we can see that there are no null values in the dataset


# In[175]:


# to obtain the description of the numerical features in the dataset
housing_data.describe()


# In[176]:


# to obtain the description of the categorical features in the dataset
housing_data.describe(include=["object"])


# In[177]:


housing_data['furnishingstatus'].value_counts()


# In[178]:


#from above output, we can see that of the 545 houses, 140 are furnished, 227 are semi-furnished and 178 are unfurnished


# In[179]:


housing_data.sort_values(by=['bedrooms','price','stories',],ascending=[False,True,True]).head()


# In[191]:


##Visualize data
#scatter plot to visualize relation between housing price and stories within house
sns.scatterplot(x=housing_data['stories'],y=housing_data['price'])


# In[192]:


#line plot to visualize relation between housing price and stories within house
sns.barplot(x=housing_data['bedrooms'],y=housing_data['price'])


# In[180]:


#replacing categorical variable with 0's and 1's for ease of use in model
housing_data=housing_data.replace(['yes','no'],[1,0])


# In[181]:


housing_data.head(5)


# In[182]:


housing_data=housing_data.replace(['furnished','semi-furnished','unfurnished'],[2,1,0])
housing_data


# In[183]:


# to obtain the correlation matrix between the different features
corr_matrix=housing_data.corr()
#from the results, area and bathrooms have the highest correlation with house price


# In[184]:


#determined the features to be considered for model from above correlation matrix
features=["area","bedrooms","bathrooms","stories","mainroad","guestroom","airconditioning","parking","prefarea","furnishingstatus"]


# In[185]:


y=housing_data.price
X = housing_data[features]


# In[186]:


# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)


# In[187]:


#model the data using Random Forest Regressor and max_features was selected using trial and error
price_model=RFR(random_state=1,oob_score=True,max_features=3)


# In[188]:


#fit the model to the training data
price_fit=price_model.fit(train_X,train_y)


# In[189]:


#predict the prices using validation subset of dataset
price_predictions=price_fit.predict(val_X)


# In[190]:


#calculate the mean absolute error, R-squared value and out-of-bag score to measure the precision of the model
price_predictions_mae=mean_absolute_error(price_predictions,val_y)
price_predictions_r2=r2_score(price_predictions,val_y)
print(f"Out-of-bag score is {price_model.oob_score_} \nMean absolute error is {price_predictions_mae} \nR squared value is {price_predictions_r2}")

