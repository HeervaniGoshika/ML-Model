# Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib


# Example dataset
data = pd.DataFrame({
    'Area': [1000, 1500, 2000, 2500, 3000],
    'Price': [100, 150, 200, 250, 300]
})

# Split features and target
X = data[['Area']]  # input
y = data['Price']   # output

# Split into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Coefficient (m):", model.coef_)
print("Intercept (c):", model.intercept_)
print("R2 Score:", r2_score(y_test, y_pred))

joblib.dump(model, "model.pkl")
