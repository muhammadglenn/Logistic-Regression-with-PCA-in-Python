# Libraries
import pandas as pd
import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt

# Dataset
heart = pd.read_excel('D:\Datasets\Kaggle\Heart Attack Analysis & Prediction Dataset\heart.xlsx')
print("Heart Dataset")
print(heart.head(10))
print("")

print("Heart Dataset")
print (heart.info())
print("")

# Change several numerical variables into category
numtocat = heart[['gender','cp','fbs','restecg','exang','slp',
                  'caa','thall','output']]
for q in numtocat:
    heart[q] = heart[q].astype('category');
print("New type of variables in dataset")
print(heart.dtypes)
print("")

# Make a list of columns
category = heart.select_dtypes('category').columns
print("categorical variables")
print(category)
print("")
numeric = heart.select_dtypes('number').columns
print("numerical variables")
print(numeric)
print("")

# Declare x and y
x = heart.drop(['output'], axis=1)
y = heart['output']

print("X variables")
print(x.head())
print("")

print("Y variables")
print(y.head())
print("")

# Split Data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)

print("X_train variables")
print(X_train.head())
print("")

print("X_test variables")
print(X_test.head())
print("")

print("Y_train variables")
print(y_train.head())
print("")

print("Y_test variables")
print(y_test.head())
print("")

# Feature Engineering
import category_encoders as ce

encoder = ce.OneHotEncoder(cols=['gender', 'cp', 'fbs', 'restecg', 'exang', 'slp', 'caa', 'thall'])
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)

print("X_train variables with dummies")
print(X_train.head())
print("")

print("X_test variables with dummies")
print(X_test.head())
print("")

# Feature Scaling
cols = X_train.columns

from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = pd.DataFrame(X_train, columns=[cols])
X_test = pd.DataFrame(X_test, columns=[cols])

print("X_train variables after scalling")
print(X_train.head())
print("")

print("X_test variables after scalling")
print(X_test.head())
print("")

# PCA
from sklearn.decomposition import PCA
pca= PCA()
pca.fit(X_train)
cumsum = np.cumsum(pca.explained_variance_ratio_)
dim = np.argmax(cumsum >= 0.90) + 1
print('The number of dimensions required to preserve 90% of variance is',dim)
print("")

# Explained variance ratio
varratio = pca.explained_variance_ratio_
print('Explained variance ratio each variables')
print(varratio)
print("")

# Plot Explained variance ratio
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlim(0,35,1)
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.show()

# Model Logistic Regression with All variables
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

print('Logistic Regression accuracy score with all variables: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

# Model Logistic Regression with 29 varibles with highest variance ratio
X_train = X_train.drop(X_train.columns[[28]],axis=1)
X_test = X_test.drop(X_test.columns[[28]],axis=1)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

print('Logistic Regression accuracy score with the first 29 features: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

# Model Logistic Regression with 28 varibles with highest variance ratio
X_train = X_train.drop(X_train.columns[[27]],axis=1)
X_test = X_test.drop(X_test.columns[[27]],axis=1)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

print('Logistic Regression accuracy score with the first 28 features: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

# Model Logistic Regression with 28 varibles with highest variance ratio
X_train = X_train.drop(X_train.columns[[26]],axis=1)
X_test = X_test.drop(X_test.columns[[26]],axis=1)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

print('Logistic Regression accuracy score with the first 27 features: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

# Model Logistic Regression with 28 varibles with highest variance ratio
X_train = X_train.drop(X_train.columns[[25]],axis=1)
X_test = X_test.drop(X_test.columns[[25]],axis=1)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

print('Logistic Regression accuracy score with the first 26 features: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))




