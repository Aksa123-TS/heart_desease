import pandas as pd
import scipy
from numpy import reshape
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

# import warnings filter
from warnings import simplefilter
df = pd.read_csv("C:/Users/SHIVA/Desktop/heart/heart/heart.csv")
print(df.head())
col = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
       'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
df.columns = col
print(df.head())
df.replace("?", np.nan, inplace=True)
print(df.isnull().sum())
df['ca'] = pd.to_numeric(df['ca'], errors='coerce')
df[['age', 'sex', 'fbs', 'exang', 'ca']] = df[['age', 'sex', 'fbs', 'exang', 'ca']].astype(int)
df[['trestbps', 'chol', 'thalach', 'oldpeak']] = df[['trestbps', 'chol', 'thalach', 'oldpeak']].astype(float)
df['target'].replace(to_replace=[1, 2, 3, 4], value=1, inplace=True)
print(df.head())
df_X = df.drop('target', axis=1)
df_y = df['target']
train_x,test_x,train_labels,test_labels=train_test_split(df_X,df_y,test_size=0.25,random_state=42)
print("shape train",train_x.shape)
print("shape test",train_labels.shape)
print("shape train",test_x.shape)
print("shape test",test_labels.shape)
simplefilter(action='ignore', category=FutureWarning)
print(test_labels)

logreg = LogisticRegression()
logreg.fit(train_x,train_labels)
y_pred=logreg.predict(test_x)
print(y_pred)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
train_x = sc_X.fit_transform(train_x)
test_x = sc_X.transform(test_x)
import sklearn.pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import KernelPCA
from imblearn.pipeline import make_pipeline

select = sklearn.feature_selection.SelectPercentile(sklearn.feature_selection.f_classif)
clf = MLPClassifier(solver='lbfgs', learning_rate='constant', activation='tanh')
kernel = KernelPCA()
    
pipeline = make_pipeline(kernel, clf)
pipeline.fit(train_x,train_labels)
t1=input("enter the number")
t2=input("enter the number")
t3=input("enter the number")
t4=input("enter the number")
t5=input("enter the number")
t6=input("enter the number")
t7=input("enter the number")
t8=input("enter the number")
t9=input("enter the number")
t10=input("enter the number")
t11=input("enter the number")
t12=input("enter the number")
t13=input("enter the number")
v = [t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13]
answer = np.array(v)
answer = answer.reshape(1,-1)
answer = sc_X.transform(answer)
print("Predicts: " + str(pipeline.predict(answer)))
