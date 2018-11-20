#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 12:17:43 2018

@author: Meng
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import seaborn as sns
from sklearn.model_selection import train_test_split

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


def header(msg):
    print('-'*60)
    print('['+ msg +']')

df= pd.read_csv('train.csv')
df_unseenX= pd.read_csv('test.csv')
#check data to see whether/what format is missing data
header('checking missing data & missing data format')
df.info() 
print(df.isnull().sum())
#Cabin has a lot of missing data,detelet
#Age has some missing, impute
#Embarked has a few missing, impute

#Next check what format is missing data, whether needs to be replaced by NaN

df.Age
#NaN, no need to replace

header('drop Cabin, PassengerId, Name, Ticket')
print('df.shape before dropping '+ str(df.shape))
df= df.drop(['Cabin', 'PassengerId','Name', 'Ticket'],axis=1)
print('df.shape after dropping '+ str(df.shape))


header('analysing features to create new features')
df['family']= df.SibSp+ df.Parch
print(df.family.describe(include=['0']))
plt.hist(df.family, bins=10)
plt.show()
print('most ppl do not have families, max=10')

header('Next let us see the relation between family# and Survived')
print(df.family)

header('Embarked VS Survived')
embark_perc = df[["Embarked", "Survived"]].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
print(embark_perc)
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(10,4))
sns.countplot('Embarked',data=df, ax=axis1, order=['S','C','Q'])
sns.barplot(x='Embarked', y='Survived', data=embark_perc, ax=axis2,order=['S','C','Q'])

header('family# VS Survived')
family_perc = df[['family', 'Survived']].groupby(['family'], as_index=False).mean().sort_values(by='Survived', ascending=False)
print(family_perc) 
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(10,4))
sns.countplot('family',data=df, ax=axis1)
sns.barplot(x='family', y='Survived', data=family_perc, ax=axis2)
#No idea whether there is a relation between family number and Survived

header('Sex VS Survived')
sex_perc=df[['Sex','Survived']].groupby(['Sex'],as_index=False).mean().sort_values(by='Survived', ascending=False)
print(sex_perc)
fig, (axis1,axis2)= plt.subplots(1,2, figsize=(10,4))
sns.countplot('Sex',data=df,ax=axis1, order=['male','female'])
sns.barplot(x='Sex',y='Survived', data=sex_perc, ax=axis2, order=['male','female'])

header('Age VS Survived: Age is continuous numerical')
fig
g=sns.FacetGrid(df, col='Survived')
g.map(plt.hist,'Age', bins=20)

header('Fare VS Survived: Fare is continuous numerical')
fig
g=sns.FacetGrid(df, col='Survived')
g.map(plt.hist, 'Fare', bins=20)

header('Pclass VS Survived: Pclass [1,2,3]')
Pclass_perc=df[['Pclass','Survived']].groupby(['Pclass'],as_index=False).mean().sort_values(by='Survived', ascending=False)
print(Pclass_perc)
fig, (axis1,axis2)= plt.subplots(1,2, figsize=(10,4))
sns.countplot('Pclass',data=df,ax=axis1)
sns.barplot(x='Pclass',y='Survived', data=Pclass_perc, ax=axis2)
print('observation: obviously class1 has higher survival rate')

header('correlating different features')
fig
g= sns.FacetGrid(df, col='Embarked', size=2.2, aspect=1.6)
g.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
g.add_legend()

header('drop SibSp, Parch')
print('df.shape after dropping '+ str(df.shape))
df= df.drop(['SibSp','Parch'],axis=1)
print('df.shape after dropping '+ str(df.shape))

header('deal with missing data..Age & Embarked')
df.Embarked.describe(include=['0'])
df.Embarked.fillna("S", inplace= True)
df.Age.fillna(df.Age.median(), inplace=True)

header('get dummies of Embarked & Sex')
df_dummies= pd.get_dummies(df, drop_first=True)
print(df_dummies.columns)
print('df_dummies.shape is '+ str(df_dummies.shape))


y= df_dummies['Survived'].values
X= df_dummies.drop('Survived',axis=1).values
#-------------Finished labled data cleansing-------------------
#                       &
#------------Start unlabled data cleansing----------------------
X_unseen= df_unseenX.drop(['Cabin', 'PassengerId','Name', 'Ticket'],axis=1) #most are null
X_unseen['family']= X_unseen.SibSp + X_unseen.Parch
X_unseen= X_unseen.drop(['SibSp','Parch'], axis=1)
unseenX_dummies= pd.get_dummies(X_unseen, drop_first=True)
unseenX_dummies.isnull().sum()
unseenX_dummies.Age.fillna(unseenX_dummies.Age.median(), inplace= True)
unseenX_dummies.Fare.fillna(unseenX_dummies.Fare.median(), inplace= True)
unseenX_dummies= unseenX_dummies.values

#-------------------Start building models------------------------ 

#split data into train, test
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.3, random_state=42)

#imp= Imputer(missing_values='NaN', strategy='mean', axis=1)

#Model: SVC

pipeline= Pipeline(steps=[('scaler', StandardScaler()),('SVM', SVC())])
param_grid = {'SVM__C':[1, 2,3,4,5, 10,20,40,80, 100],
              'SVM__gamma':[0.6,0.5,0.4,0.3, 0.2, 0.1, 0.05, 0.01]}

cv = GridSearchCV(pipeline, param_grid,cv=3)
cv.fit(X_train,y_train)

header('SVC result')
print(cv.best_params_)
print(cv.score(X_train,y_train))
print(cv.score(X_test, y_test))
y_pred= cv.predict(X_test)
print(classification_report(y_test,y_pred))

#Model: Logistic Regression
logreg = LogisticRegression()
c_space = np.logspace(-6, 8, 15)
param_grid = {'logistic__C': c_space}
pipeline= Pipeline(steps=[('scaler', StandardScaler()),('logistic', logreg)])
cv = GridSearchCV(pipeline, param_grid,cv=3)
cv.fit(X_train,y_train)
header('Logistic Regression result')
print(cv.best_params_)
print(cv.score(X_train,y_train))
print(cv.score(X_test, y_test))
y_pred= cv.predict(X_test)
print(classification_report(y_test,y_pred))



# Random Forests

random_forest = RandomForestClassifier()
param_grid = { 
    'rnf__n_estimators': [200, 500],
    'rnf__max_features': ['auto', 'sqrt', 'log2'],
    'rnf__max_depth' : [4,5,6,7,8],
    'rnf__criterion' :['gini', 'entropy']
}
pipeline= Pipeline(steps=[('scaler', StandardScaler()),('rnf', random_forest)])
cv = GridSearchCV(pipeline, param_grid,cv=3)
cv.fit(X_train,y_train)
header('Random forest result')
print(cv.best_params_)
print(cv.score(X_train,y_train))
print(cv.score(X_test, y_test))
y_pred= cv.predict(X_test)
print(classification_report(y_test,y_pred))


# KNeighbors

knn= KNeighborsClassifier()
param_grid={'knn__n_neighbors': np.linspace(1,10,10).astype(int)}
pipeline= Pipeline(steps=[('scaler', StandardScaler()),('knn', knn)])
cv = GridSearchCV(pipeline, param_grid,cv=3)
cv.fit(X_train,y_train)
header('KNeighborsClassifier result')
print(cv.best_params_)
print(cv.score(X_train,y_train))
print(cv.score(X_test, y_test))
y_pred= cv.predict(X_test)
print(classification_report(y_test,y_pred))


# Gaussian Naive Bayes

gaussian = GaussianNB()
gaussian.fit(X_train, y_train)
header('Gaussian Naive Bayes result')
print(gaussian.score(X_train,y_train))
print(gaussian.score(X_test, y_test))
y_pred= gaussian.predict(X_test)
print(classification_report(y_test,y_pred))
















#predicting unseen data
y_pred_unseen= cv.predict(unseenX_dummies)

submission = pd.DataFrame({
        "PassengerId": df_unseenX["PassengerId"],
        "Survived": y_pred_unseen
    })
submission.to_csv('titanic.csv', index=False)








#X_train= df_train.drop('Survived',axis=1).values

