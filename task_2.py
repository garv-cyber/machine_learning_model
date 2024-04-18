# importing the required modules
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# loading the dataset from a CSV file into a DataFrame object named 'df'
df = pd.read_csv("IMDb Movies India.csv", encoding='latin1')

# dropping the NaN values from the dataframe
df.dropna(inplace=True)   

# Encode categorical variables
encoder = OneHotEncoder()
ct = ColumnTransformer([('encoder', encoder, ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3'])], remainder='passthrough')

# splitting the data into X and y 
X = ct.fit_transform(df[['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']])
y = df['Rating']

# Splitting the data into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# Model Training
lr = LinearRegression()
lr.fit(X_train, y_train)

# Evaluating the model and making predictions
print("Evaluating the Linear Regression Model Performance...")

y_train_pred = lr.predict(X_train)   # Predicting the rating for train data
y_pred = lr.predict(X_test)          # Predicting the rating for test data

lr_train_mse = mean_squared_error(y_train, y_train_pred)  # Calculate MSE for training set
lr_test_mse = mean_squared_error(y_test, y_pred)          # Calculate MSE for test set

# Calculate R-squared score
r2 = r2_score(y_test, y_pred)

# Creating the dataframe to show the results
lr_results = pd.DataFrame([['Linear Regression', lr_train_mse, lr_test_mse, r2]], columns=['Method', 'Train MSE', 'Test MSE', 'R-squared'])
print(lr_results)