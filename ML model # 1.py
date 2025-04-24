#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 


# In[31]:


#df = pd.read_csv("C:/Users/subba/OneDrive/Desktop/movie/iris-dataset.csv")


# In[32]:


#df


# In[20]:


#df.size


# In[21]:


#fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10, 4))
sepal_length = df.sepal_length
sepal_width = df.sepal_width
petal_length = df.petal_length
petal_width	 = df.petal_width
#flower = df.flower
plt.scatter(flower,sepal_length, marker = "*",color = 'red')
plt.scatter(flower,sepal_width, marker = "*",color = 'black')
plt.scatter(flower,petal_length, marker = "*",color = 'yellow')
plt.scatter(flower,petal_width, marker = "*",color = 'blue')
plt.grid(axis='y',color = "grey")
#plt.grid(axis='x',color = "grey")


# In[22]:


#plt.scatter(a,b,marker="*",color = "red")
#plt.grid(axis = 'y')
#plt.grid(axis = 'x')
#plt.scatter(c,d,marker="*",color = "Blue")


# In[23]:


#we use scatter plot in order to know the relationship between data points 
#why not histo or pie or bar because we are not comparing and we cannot make accurate predictions than scatter plotting techniques 
#we are goint to use SVM (SUPPORT VECTOR MACHINE) because it is easy to handle simple data and makes accurate predictions 
#why not knn or other models because 1. KNN is also applicable but used more for remembering or storing the data pounts than traning them 
#                                    2. Logistic regression sometimes gives overfitting cause data is not small enough 
#                                    3. Decition tree is one of hte best choice but it needs more data to shine 


# In[24]:


# lets see the percentage of pridiction on the iris data set through all the models
# 1. knn 2. logistic regression 3.support vector machine 4.naive baise


# In[17]:


#************* SVM ***************#


# In[2]:


import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd
iris = load_iris()


# In[3]:


iris.feature_names


# In[4]:


df = pd.DataFrame(iris.data,columns=iris.feature_names)
df.head()


# In[5]:


df['target'] = iris.target
df.head()


# In[6]:


df[df.target==1].head()


# In[7]:


df['Flower_name'] = df.target.apply(lambda x : iris.target_names[x])
df.head()


# In[8]:


df0=df[df.target==0]
df1=df[df.target==1]
df2=df[df.target==2]
df1.head()


# In[9]:


plt.scatter(df0['sepal length (cm)'],df0['sepal width (cm)'],marker='+')
plt.scatter(df1['sepal length (cm)'],df1['sepal width (cm)'],marker='+')


# In[ ]:





# In[32]:


X = df.drop(['target','Flower_name'], axis='columns') # beacuse we dont need target and flower_name
y = df.target
df.tail()


# In[33]:


x=[6.5,3.0,5.2,2.0]


# In[11]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size = 0.2 )


# In[12]:


len(X_test)


# In[13]:


from sklearn.svm import SVC
model = SVC(C=10)# we can change value of C -->Regularization strength and many more functions to increase model performance 


# In[14]:


model.fit(X_train,Y_train)


# In[15]:


model.score(X_test, Y_test)


# In[16]:


model.score(X_test,Y_test)


# In[17]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
# 1. Make predictions
y_pred = model.predict(X_test)
# 2. Create confusion matrix
cm = confusion_matrix(Y_test, y_pred)
# 3. Plot it as a heatmap
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.show()


# In[18]:


labels = iris.target_names
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)


# In[19]:


#************KNN******************#


# In[37]:


from sklearn.neighbors import KNeighborsClassifier
knn =  KNeighborsClassifier(n_neighbors=15)


# In[ ]:





# In[38]:


knn.fit(X_train,Y_train)


# In[39]:


knn.score(X_train,Y_train)


# In[40]:


knn.score(X_test,Y_test)


# In[29]:


X_train


# In[25]:


X_test


# In[ ]:





# In[ ]:





# In[44]:


import numpy as np 
def predict_iris(sepal_length, sepal_width, petal_length, petal_width):
    sample = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = knn.predict(sample)
    class_name = iris.target_names[prediction[0]]
    return class_name

# Example usage
output = predict_iris(6.5,3.0,5.2,2.0)
print("Predicted Iris species:", output)


# In[ ]:





# In[35]:


x


# In[ ]:


sample1 = np.arr([[sepal_length, sepal_width, petal_length, petal_width]])
predict = knn.pridict(sample1)

