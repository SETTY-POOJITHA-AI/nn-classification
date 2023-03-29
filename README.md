# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

![neural network](https://user-images.githubusercontent.com/93427581/228508789-2f13465e-b2df-4035-8a02-93c69669ce75.png)


## DESIGN STEPS

### STEP 1:
Import the necessary packages & modules

### STEP 2:
Load and read the dataset

### STEP 3:
Perform pre processing and clean the dataset

### Step 4:
Encode categorical value into numerical values using ordinal/label/one hot encoding

### Step 5:
Visualize the data using different plots in seaborn

### Step 6:
Normalize the values and split the values for x and y

### Step 7:
Build the deep learning model with appropriate layers and depth

### Step 8:
Analyze the model using different metrics

### Step 9:
Plot a graph for Training Loss, Validation Loss Vs Iteration & for Accuracy, Validation Accuracy vs Iteration

### Step 10:
Save the model using pickle

###Step 11:
Using the DL model predict for some random inputs

## PROGRAM
```
Developed by: Poojitha.S
Register Num: 212221240050
```
### Import Libraries
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder

from sklearn.metrics import classification_report as report
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import confusion_matrix as conf

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
```
### Read the data & Pre Process
```
df = pd.read_csv("./customers.csv")

df.columns
df.dtypes
df.shape
df.isnull().sum()

df = df.drop('ID',axis=1)
df = df.drop('Var_1',axis=1)

df_cleaned = df.dropna(axis=0)

df_cleaned.isnull().sum()
df_cleaned.shape
df_cleaned.dtypes
```
### Encoding categorical values
```
df_cleaned['Gender'].unique()
df_cleaned['Ever_Married'].unique()  
df_cleaned['Graduated'].unique()
df_cleaned['Profession'].unique()
df_cleaned['Spending_Score'].unique()
df_cleaned['Segmentation'].unique()


categories_list=[['Male', 'Female'],['No', 'Yes'],
                 ['No', 'Yes'],['Healthcare', 'Engineer',
                 'Lawyer','Artist', 'Doctor','Homemaker',
                 'Entertainment', 'Marketing', 'Executive'],
                 ['Low', 'Average', 'High']]

enc = OrdinalEncoder(categories=categories_list)

df1 = df_cleaned.copy()

df1[['Gender','Ever_Married',
     'Graduated','Profession',
     'Spending_Score']] = enc.fit_transform(df1[['Gender',
     						'Ever_Married','Graduated',
                            'Profession','Spending_Score']])
df1
df1.dtypes

le = LabelEncoder()
df1['Segmentation'] = le.fit_transform(df1['Segmentation'])

df1.dtypes
```
### Data Visualization
```
corr = df1.corr()

sns.heatmap(corr, 
            xticklabels=corr.columns,
            yticklabels=corr.columns,
            cmap="BuPu",
            annot= True)

sns.distplot(df1['Age'])

plt.figure(figsize=(10,6))
sns.scatterplot(x='Family_Size',y='Age',data=df1)
```
### Assign X and Y values
```
scale = MinMaxScaler()
scale.fit(df1[["Age"]]) # Fetching Age column alone
df1[["Age"]] = scale.transform(df1[["Age"]])

df1.describe()

df1['Segmentation'].unique()

x = df1[['Gender','Ever_Married','Age','Graduated',
		 'Profession','Work_Experience','Spending_Score',
         'Family_Size']].values
         
y1 = df1[['Segmentation']].values

ohe = OneHotEncoder()
ohe.fit(y1)

y = ohe.transform(y1).toarray()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=1/3,random_state=50)
```
### Building Model
```
ai = Sequential([Dense(50,input_shape = [8]),
                 Dense(40,activation="relu"),
                 Dense(30,activation="relu"),
                 Dense(20,activation="relu"),
                 Dense(4,activation="softmax")])

ai.compile(optimizer='adam',
           loss='categorical_crossentropy',
           metrics=['accuracy'])

early_stop = EarlyStopping(
    monitor='val_loss',
    mode='max', 
    verbose=1, 
    patience=20)
    
ai.fit( x = x_train, y = y_train,
        epochs=500, batch_size=256,
        validation_data=(x_test,y_test),
        callbacks = [early_stop]
        )
```
### Analyze the model - Metrics
```
metrics = pd.DataFrame(ai.history.history)
metrics.head()

metrics[['loss','val_loss']].plot()

metrics[['accuracy','val_accuracy']].plot()

x_pred = np.argmax(ai.predict(x_test), axis=1)
x_pred.shape

y_truevalue = np.argmax(y_test,axis=1)
y_truevalue.shape

conf(y_truevalue,x_pred)

print(report(y_truevalue,x_pred))
```
### Saving the model
```
import pickle

# Saving the Model
ai.save('customer_classification_model.h5')
     
# Saving the data
with open('customer_data.pickle', 'wb') as fh:
   pickle.dump([x_train,y_train,x_test,y_test,df1,df_cleaned,scale,enc,ohe,le], fh)
     
# Loading the Model
ai_brain = load_model('customer_classification_model.h5')
     
# Loading the data
with open('customer_data.pickle', 'rb') as fh:
   [x_train,y_train,x_test,y_test,df1,df_cleaned,scale,enc,ohe,le]=pickle.load(fh)
   ```
### Sample Predicition
```
x_prediction = np.argmax(ai_brain.predict(x_test[1:2,:]), axis=1)

print(x_prediction)

print(le.inverse_transform(x_prediction))
```
## Dataset Information
![database information](https://user-images.githubusercontent.com/93427581/228511419-cc3d3385-28ad-4d1f-ab3e-065a22421a53.png)

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot
![graph](https://user-images.githubusercontent.com/93427581/228512177-e97c9d66-1d02-404c-b5ad-798997ed8556.png)



### Classification Report

![classification report](https://user-images.githubusercontent.com/93427581/228512248-d78f9fb2-3eae-4ae0-99a6-0044a2a44be8.png)


### Confusion Matrix

![confusion matrix](https://user-images.githubusercontent.com/93427581/228512293-052735b0-8f8f-4906-a481-37f6875a0be7.png)



### New Sample Data Prediction

![new sample](https://user-images.githubusercontent.com/93427581/228512419-4a43e8da-eb24-44e2-a2c6-a89e1620f334.png)


## RESULT
A neural network classification model is developed for the given dataset.
