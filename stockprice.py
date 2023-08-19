import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset (assuming the dataset has 'Date', 'Open', 'High', 'Low', 'Close', 'Volume')
data = pd.read_csv('stock_data.csv')  # Replace with your dataset file

# Choose the features for prediction (e.g., 'Open', 'High', 'Low', 'Volume')
features = ['Open', 'High', 'Low', 'Volume']
target = 'Close'

# Create lag features
look_back = 30  # Number of previous days to consider
for feature in features:
    for i in range(1, look_back + 1):
        data[f'{feature}_lag_{i}'] = data[feature].shift(i)

# Drop rows with missing values
data.dropna(inplace=True)

# Select features and target
X = data.drop(target, axis=1)
y = data[target]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
