import pandas as pd
from numpy import reshape
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import KernelPCA
from imblearn.pipeline import make_pipeline
from warnings import simplefilter  # import warnings_filter
#from sklearn.feature_selection import RFE
#from sklearn import model_selection
#from sklearn.model_selection import cross_val_score
#from sklearn import metrics


df = pd.read_csv("./heart.csv")  # read the data file
col = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
       'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

# take the columns as a list
df.columns = col 

df.replace("?", np.nan, inplace=True)
# Convert ca values to a numeric type. Using coerce can set invalid parsing will be as NaN
df['ca'] = pd.to_numeric(df['ca'], errors='coerce') 

# Convert all data to int and float format
df[['age', 'sex', 'fbs', 'exang','ca']].astype(int)
df[['trestbps', 'chol', 'thalach', 'oldpeak']].astype(float)

# target values contain 1,2,3,4 so ,replacing these values into 1, so that target must contain only 1 and 0
df['target'].replace(to_replace=[1, 2, 3, 4], value=1, inplace=True)


# df_x contains data without target data  and df_y contains only target data .After that, split it into 25% test and 75% train data

df_X = df.drop('target', axis=1)
df_y = df['target']
train_x, test_x, train_labels, test_labels = train_test_split(df_X, df_y, test_size=0.25, random_state= 42)

simplefilter(action='ignore', category=FutureWarning)  # ignore future warning

logreg = LogisticRegression()       # use logistic regression because target has 0 and  1 
logreg.fit(train_x, train_labels)   # generalizes to similar data to that on which it was trained.
y_pred = logreg.predict(test_x)     # predict data and store the predicted data in y_pred


sc_X = StandardScaler()  # Standardize features by removing the mean and scaling to unit variance

# Fit to data, then "transform" it with the previously computed mean and std to autoscale the data (subtract mean from all values and then divide it by std).
train_x = sc_X.fit_transform(train_x)
test_x = sc_X.transform(test_x)

# ‘tanh’: the hyperbolic tan function, returns f(x) = tanh(x),‘lbfgs’ :is an optimizer in the family of quasi-Newton methods,Constant learning rates are always smaller than 1 
clf = MLPClassifier(solver='lbfgs', learning_rate='constant', activation='tanh')

kernel = KernelPCA()#Non-linear dimensionality reduction through the use of kernels.

# The pipeline object is in the form of (key, value) pairs. Key is a string that has the name for a particular step and value is the name of the function or actual method. 

pipeline = make_pipeline(kernel, clf)
pipeline.fit(train_x, train_labels)

# setting some default values 
age = 59
sex = 1  # female=1 male=0
cp = 2
trestbps = 150
chol = 212
fbs = 1
restecg = 1
thalach = 157
exang = 0
oldpeak = 1.6
slope = 2
ca = 0
thal = 2 

v = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]  # Creating a list using these values
answer = np.array(v)  # Convert it as array
answer = answer.reshape(1, -1)  # Reshape the array to 2 dimensional format
answer = sc_X.transform(answer) # Normalize the data as the training data format

print("Predicts: " + str(pipeline.predict(answer)))  # Finaly predict and print the result
