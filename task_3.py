# Importing the required libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Importing the dataset
df = pd.read_csv("IRIS.csv")

# Encoding categorical target variable
le = LabelEncoder()
df['species'] = le.fit_transform(df['species'])

# Splitting the dataset into features (X) and target (y)
X = df.drop('species', axis=1)  # Features
y = df['species']  # Target

# Splitting the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Building and training the logistic regression model
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)

# Making predictions
y_pred = log_reg.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Plotting the graph (scatter plot for two features)
plt.scatter(X_test[y_test == 0]['sepal_length'], X_test[y_test == 0]['petal_length'], label='Setosa', color='red')
plt.scatter(X_test[y_test == 1]['sepal_length'], X_test[y_test == 1]['petal_length'], label='Versicolor', color='blue')
plt.scatter(X_test[y_test == 2]['sepal_length'], X_test[y_test == 2]['petal_length'], label='Virginica', color='green')

plt.xlabel('Sepal Length')
plt.ylabel('Petal Length')
plt.title('Iris Species Classification')
plt.legend()
plt.show()