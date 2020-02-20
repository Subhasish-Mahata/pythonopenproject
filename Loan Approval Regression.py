#!/usr/bin/env python
# coding: utf-8

# # import libraries

# In[79]:


import pandas as pd
import numpy as  np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


# # Import Data Set

# In[80]:


df=pd.read_csv("C:\\Users\\Anand\\Desktop\\Academic\\data science\\Mini Projects\\Loan Approval - Regression\\Dataset\\Train.csv")

test=pd.read_csv("C:\\Users\\Anand\\Desktop\\Academic\\data science\\Mini Projects\\Loan Approval - Regression\\Dataset\\Test.csv")
train_original = df.copy()
test_original = test.copy()


# # Descriptive

# In[81]:


df.head()


# In[82]:


df.shape


# In[83]:


df.columns


# In[84]:


df.Loan_Status.unique()


# In[85]:


encode = LabelEncoder()
df.Loan_Status = encode.fit_transform(df.Loan_Status)
df.head()


# # Missing Values Handelling

# In[86]:


# check for missing values
df.isnull().sum()


# In[87]:


# replace missing values with the mode
df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['Married'].fillna(df['Married'].mode()[0], inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)

df['Loan_Amount_Term'].value_counts()

df.head()


# In[88]:


# replace missing value with the mode
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
df.head()


# In[89]:


# replace missing values with the median value due to outliers
df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
df.head()


# In[90]:


df.isnull().sum()


# # Visualization

# In[91]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[92]:


sns.barplot(x=df['Gender'], y = df['Loan_Status'],  
            data = df, label="Relationship among Gender and Loan Approval Status", ci=None)


# In[93]:


sns.barplot(x=df["Married"], y = df['Loan_Status'],  
            data = df, label="Relationship among Gender and Loan Approval Status", ci=None)


# In[94]:


sns.catplot(x="Married", y="ApplicantIncome", hue="Loan_Status",
            col="Gender", aspect=.9,
            kind="swarm", data=df);


# In[95]:


sns.catplot(x="Married", y="ApplicantIncome", hue="Loan_Status",
            col="Education", aspect=.6,
            kind="swarm", data=df);


# In[96]:


sns.barplot(x=df["Credit_History"], y = df['Loan_Status'],  
            data = df, label="Relationship among Credit_History and Loan Approval Status", ci=None)


# # Model Building

# In[97]:


# replace missing values in Test set with mode/median from Training set
test['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
test['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
test['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
test['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)
test['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
test['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
test.isnull().sum()
test.head()


# In[98]:


# before log transformation
ax1 = plt.subplot(121)
df['LoanAmount'].hist(bins=20, figsize=(12,4))
ax1.set_title("Train")

ax2 = plt.subplot(122)
test['LoanAmount'].hist(bins=20)
ax2.set_title("Test")


# In[99]:


# Removing skewness in LoanAmount variable by log transformation
df['LoanAmount_log'] = np.log(df['LoanAmount'])
test['LoanAmount_log'] = np.log(test['LoanAmount'])
test.head()


# In[100]:


# after log transformation
ax1 = plt.subplot(121)
df['LoanAmount_log'].hist(bins=20, figsize=(12,4))
ax1.set_title("Train")

ax2 = plt.subplot(122)
test['LoanAmount_log'].hist(bins=20)
ax2.set_title("Test")


# In[101]:


# drop Loan_ID 
df = df.drop('Loan_ID', axis=1)
test = test.drop('Loan_ID', axis=1)
df.columns


# In[102]:


# drop "Loan_Status" and assign it to target variable
y = df.Loan_Status
X = df.drop('Loan_Status', 1)
y.head()
X.head()


# In[103]:


# adding dummies to the dataset
X = pd.get_dummies(X)
df = pd.get_dummies(df)
test = pd.get_dummies(test)
test.head()



# In[104]:


X.shape, df.shape, test.shape


# In[105]:


# split the data into train and cross validation set
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)


# In[106]:



model=LogisticRegression()
model.fit(x_train,y_train)


# In[107]:


# make prediction
pred_test = model.predict(x_test)


# In[110]:


pred_test


# # Model Evaluation

# In[111]:


# calculate accuracy score
from sklearn.metrics import accuracy_score
score=accuracy_score(y_test,pred_test)
score


# In[112]:


# import confusion_matrix
from sklearn.metrics import confusion_matrix
 
cm = confusion_matrix(y_test, pred_test)
print(cm)

# ploting
sns.heatmap(cm, annot=True, fmt="d")
plt.title('Confusion matrix')
plt.xlabel('Predicted')
plt.ylabel('True')


# In[113]:


# Precision , Recall 
from sklearn.metrics import classification_report
print(classification_report(y_test, pred_test))


# # Final Precdiction on Test Data

# In[117]:


prediction_test = model.predict(test)


# # Store Result

# In[118]:


Submission=pd.read_csv("C:\\Users\\Anand\\Desktop\\Academic\\data science\\Mini Projects\\Loan Approval - Regression\\Dataset\\Submission.csv")


# In[120]:


Submission['Loan_Status'] = prediction_test
Submission['Loan_ID'] = test_original['Loan_ID']


# In[118]:


#Let's Replace 1 with Yes and 0 with NO


# In[122]:


Submission['Loan_Status'].replace(0, 'NO', inplace=True)
Submission['Loan_Status'].replace(1, 'Yes', inplace=True)


# In[123]:


Submission.head()


# In[127]:


# convert - CSV file
Submission.to_csv('LoanApr.csv', index=False)


# In[ ]:




