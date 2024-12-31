import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Read the dataset
data = pd.read_csv('driving_dataset.csv')

# Print the original dataset
print("Original Dataset:")
print(data)

# Step 2: Extract independent and dependent variables
x = data.iloc[:, 0].values.reshape(-1, 1)  # Independent variable ('hours')
y = data.iloc[:, 1].values  # Dependent variable ('risk_score')

# Print extracted variables
print("\nIndependent variable (x):")
print(x)

print("\nDependent variable (y):")
print(y)

# Step 3: Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Print training and test data
print("\nTraining Data:")
print(pd.DataFrame({'Hours': x_train.flatten(), 'Risk Score': y_train}))

print("\nTest Data:")
print(pd.DataFrame({'Hours': x_test.flatten(), 'Risk Score': y_test}))

# Step 4: Create and fit the linear regression model
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Step 5: Predict on the test set
y_pred = regressor.predict(x_test)

# Print regression coefficients
print(f"\nRegression Coefficient (Slope): {regressor.coef_[0]:.3f}")
print(f"Intercept: {regressor.intercept_:.3f}")

# Print predictions
print("\nPredictions on Test Data:")
predictions_df = pd.DataFrame({'Hours': x_test.flatten(), 'Actual Risk Score': y_test, 'Predicted Risk Score': y_pred})
print(predictions_df)

# Step 7: Visualizing the Training and Test Set Results
plt.figure(figsize=(12, 6))

# Scatter plot of the training set results
plt.scatter(x_train, y_train, color='purple', label='Training data')

# Plot the regression line for the training set
plt.plot(x_train, regressor.predict(x_train), color='cyan', label='Regression Line (Training Set)')

# Scatter plot of the test set results
plt.scatter(x_test, y_test, color='orange', label='Test data')

# Plot the regression line for the test set
plt.plot(x_test, y_pred, color='magenta', linestyle='--', label='Regression Line (Test Set)')

plt.title('Training and Test Set Results')
plt.xlabel('Hours Spent Driving')
plt.ylabel('Risk Score')
plt.legend()
plt.grid(True)
plt.show()
