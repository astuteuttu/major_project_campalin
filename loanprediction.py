# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv('loan_data.csv')  # Replace with your dataset file

# Data preprocessing
# Assuming 'marital_status', 'education', 'dependents', 'employment' are columns in the dataset
X = data[['marital_status', 'education', 'dependents', 'employment']]
y = data['loan_amount']

# Convert categorical variables to numerical using one-hot encoding
X = pd.get_dummies(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predict loan amounts on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# You can now use the trained model to make predictions for new user inputs
# Example: New user inputs in the same format as X_test_scaled
new_user_inputs = scaler.transform([[1, 2, 3, 1]])  # Modify the values accordingly
predicted_loan_amount = model.predict(new_user_inputs)
print(f"Predicted Loan Amount for New User: {predicted_loan_amount}")
