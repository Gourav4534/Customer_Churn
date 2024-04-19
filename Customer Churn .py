#!/usr/bin/env python
# coding: utf-8

# ## Customer_churn Dataset
# 
# **Tasks To Be Performed:**
# 
# **1. Data Manipulation**

# In[2]:


# Importing Library 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# In[3]:


# import data
data = pd.read_csv('customer_churn.csv')


# In[4]:


data.head()


# In[5]:


data.shape


# In[6]:


# Extract the 5th column and store it in ‘customer_5’
customer_5 = data.iloc[:,4]
customer_5.head()


# In[7]:


# Extract the 15th column and store it in ‘customer_15’
customer_15 = data.iloc[:,14]
customer_15.head()


# In[8]:


#  Extract all the male senior citizens whose payment method is electronic check and store the result in ‘senior_male_electronic’

senior_male_electronic =  data[(data['gender']=='Male') & (data['PaymentMethod'] =='Electronic check')]
senior_male_electronic.head()


# In[9]:


# Extract all those customers whose tenure is greater than 70 months or
#their monthly charges is more than $100 and store the result in
#‘customer_total_tenure’

customer_total_tenure = data[(data['tenure'] > 70)  | (data['MonthlyCharges']>100)]
customer_total_tenure.head()


# In[10]:


# Extract all the customers whose contract is of two years, payment method
#is mailed check and the value of churn is ‘Yes’ and store the result in
#‘two_mail_yes’

two_mail_yes = data[(data['Contract']== 'Two year' ) & (data['PaymentMethod']=='Mailed check') & (data['Churn']=='Yes')]
two_mail_yes.head()


# In[11]:


# Extract 333 random records from the customer_churndataframe and store
# the result in ‘customer_333’

customer_333 = data.sample(n=333)
print(customer_333.shape)
customer_333.head()


# In[12]:


# Get the count of different levels from the ‘Churn’ column
data['Churn'].value_counts()


# **2. Data Visualization**

# In[13]:


# Build a bar-plot for the ’InternetService’ column:
# a. Set x-axis label to ‘Categories of Internet Service’
# b. Set y-axis label to ‘Count of Categories’
# c. Set the title of plot to be ‘Distribution of Internet Service’
# d. Set the color of the bars to be ‘orange

x= data['InternetService'].value_counts().keys()
y= data['InternetService'].value_counts()
plt.bar(x,y,color='orange')
plt.xlabel('Categories of Internet Service')
plt.ylabel('Count of Categories')
plt.title('Distribution of Internet Service')
plt.show()


# In[14]:


# Build a histogram for the ‘tenure’ column:
#a. Set the number of bins to be 30
#b. Set the color of the bins to be ‘green’
#c. Assign the title ‘Distribution of tenure'

plt.hist(data['tenure'],color='green',bins=30)
plt.title('Distribution of tenure')


# In[15]:


# Build a scatter-plot between ‘MonthlyCharges’ and ‘tenure’. Map ‘MonthlyCharges’ to the y-axis and ‘tenure’ to the ‘x-axis’:
#a. Assign the points a color of ‘brown’
#b. Set the x-axis label to ‘Tenure of customer’
#c. Set the y-axis label to ‘Monthly Charges of customer’
#d. Set the title to ‘Tenure vs Monthly Charges’
#e. Build a box-plot between ‘tenure’ & ‘Contract’. Map ‘tenure’ on they-axis &
# f. ‘Contract’ on the x-axis

plt.figure(figsize=(10,5))
plt.scatter(x=data['tenure'],y=data['MonthlyCharges'],color='brown')
plt.xlabel('Tenure of customer')
plt.ylabel('Monthly Charges of customer')
plt.title('Tenure vs Monthly Charges')
plt.show()

# box plot
data.boxplot(column='tenure',by='Contract')
plt.show()


# **3. Linear Regression**

# In[16]:


#Build a simple linear model where dependent variable is ‘MonthlyCharges’ and independent variable is ‘tenure’:
#a. Divide the dataset into train and test sets in 70:30 ratio.
#b. Build the model on train set and predict the values on test set
#c. After predicting the values, find the root mean square error
#d. Find out the error in prediction & store the result in ‘error’
# e. Find the root mean square error

from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

x= pd.DataFrame(data['tenure']) #independent Variable
y=pd.DataFrame(data['MonthlyCharges']) #dependent Variable
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.30,random_state=0)


# In[17]:


x_train.shape, x_test.shape, y_train.shape, y_test.shape


# In[18]:


regressor = LinearRegression()
regressor.fit(x_train,y_train)


# In[19]:


y_pred=regressor.predict(x_test)
y_pred[:5],y_test[:5]


# In[20]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test,y_pred)
rmse=np.sqrt(mse)   #lower the value beter the model is
print(rmse)


# **3.Logistic Regression**

# In[21]:


# Build a simple logistic regression model where dependent variable is‘Churn’ and independent variable is ‘MonthlyCharges’:
#a. Divide the dataset in 65:35 ratio
#b. Build the model on train set and predict the values on test set
#c. Build the confusion matrix and get the accuracy score
#d. Build a multiple logistic regression model where dependent variable is ‘Churn’ and independent variables are ‘tenure’ and ‘MonthlyCharges’
#e. Divide the dataset in 80:20 ratio
#f. Build the model on train set and predict the values on test set
#g. Build the confusion matrix and get the accuracy score

from sklearn.linear_model import LogisticRegression


x= pd.DataFrame(data['MonthlyCharges'])
y= pd.DataFrame(data['Churn'])
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.35,random_state=0)


# In[22]:


x_train.shape, x_test.shape, y_train.shape, y_test.shape


# In[23]:


logic = LogisticRegression()
logic.fit(x_train,y_train)


# In[24]:


y_pred = logic.predict(x_test)


# In[25]:


from sklearn.metrics import accuracy_score,confusion_matrix
conf_matrix=confusion_matrix(y_test,y_pred)
accuracy=accuracy_score(y_test,y_pred)

print("Logistic Regression")
print("Confusion Matrix:")
print(conf_matrix)
print("Accuracy Score:", accuracy)


# In[26]:


from sklearn.linear_model import LogisticRegression


x= pd.DataFrame(data[['MonthlyCharges','tenure']])
y= pd.DataFrame(data['Churn'])
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.20,random_state=0)


# In[27]:


x_train.shape, x_test.shape, y_train.shape, y_test.shape


# In[28]:


logic_multiple = LogisticRegression()
logic_multiple.fit(x_train,y_train)


# In[29]:


y_pred = logic_multiple.predict(x_test)


# In[30]:


from sklearn.metrics import accuracy_score,confusion_matrix
conf_matrix=confusion_matrix(y_test,y_pred)
accuracy=accuracy_score(y_test,y_pred)

print("Logistic Regression With Multiple Independent features")
print("Confusion Matrix:")
print(conf_matrix)
print("Accuracy Score:", accuracy)


# **5. Decision Tree**

# In[31]:


#Build a decision tree model where dependent variable is ‘Churn’ and independent variable is ‘tenure’:
# a. Divide the dataset in 80:20 ratio
# b. Build the model on train set and predict the values on test set
# c. Build the confusion matrix and calculate the accuracy

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

x= pd.DataFrame(data['tenure'])
y= pd.DataFrame(data['Churn'])
x_train,x_test,y_train, y_test= train_test_split(x,y,test_size=0.20,random_state=0)


# In[32]:


decision_tree = DecisionTreeClassifier()
decision_tree.fit(x_train,y_train)
y_pred = decision_tree.predict(x_test)


# In[33]:


conf_matrix= confusion_matrix(y_test,y_pred)
accuracy = accuracy_score(y_test,y_pred)

print("Decision Tree:")
print("Confusion Matrix:")
print(conf_matrix)
print("Accuracy Score:", accuracy)


# **6. Random Forest**

# In[34]:


#Build a Random Forest model where dependent variable is ‘Churn’ and
#independent variables are ‘tenure’ and ‘MonthlyCharges’:
#a. Divide the dataset in 70:30 ratio
#b. Build the model on train set and predict the values on test set
#c. Build the confusion matrix and calculate the accuracy

from sklearn.ensemble import RandomForestClassifier


x = data[['tenure', 'MonthlyCharges']]
y = data['Churn']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=0)


# In[35]:


random_forest = RandomForestClassifier()
random_forest.fit(x_train,y_train)
y_pred = random_forest.predict(x_test)


# In[36]:


conf_matrix = confusion_matrix(y_test,y_pred)
accuracy = accuracy_score(y_test,y_pred)

print("Random Forest:")
print("Confusion Matrix:")
print(conf_matrix)
print("Accuracy Score:", accuracy)


# In[ ]:





# In[ ]:





# In[ ]:




