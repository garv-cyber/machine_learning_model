# importing the required modules
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import  mean_squared_error

# importing the data set
df = pd.read_csv("IRIS.csv")

# doing preprocessing
df.dropna(inplace=True)     # removing the nan values if any
le = LabelEncoder()
df['species'] = le.fit_transform(df['species'])

# splipting the data into train and test data
x = df.drop('species', axis = 1)
y = df['species']

# spliting the data into train and test set
X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size=0.2, random_state=100)

# building model
lr = LinearRegression()
lr.fit(X_train, Y_train)

# Applying the model to make prediction
y_lr_train_pred = lr.predict(X_train)
y_lr_test_pred = lr.predict(X_test)

# Evaluate the linear regression model performance
print("Evaluating the Linear Regression model performance...")

# for training set
lr_train_mse = mean_squared_error(Y_train, y_lr_train_pred)

# for test set
lr_test_mse = mean_squared_error(Y_test, y_lr_test_pred)

# printing the results in the form of a dataset
lr_results = pd.DataFrame(['Linear Regression',lr_train_mse,lr_test_mse]).transpose()
lr_results.columns = ['Method','Training MSE','Test MSE']
print(lr_results)