# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv('housing_data.csv')  # Replace with your dataset file

# Data preprocessing
X = data.drop('price', axis=1)  # Features
y = data['price']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predict house prices on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# You can now use the trained model to predict prices for new data
# Example: New data in the same format as X_test_scaled
new_data = scaler.transform([[4, 0.3, 8, 1, 0.5, ...]])  # Modify the values accordingly
predicted_price = model.predict(new_data)
print(f"Predicted Price for New Data: {predicted_price}")
