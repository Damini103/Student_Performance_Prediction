import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================
# 1. Load Dataset
# ==========================
data = pd.read_csv("Dataset/StudentsPerformance.csv")

print("Dataset Loaded Successfully\n")

# ==========================
# 2. Basic Information
# ==========================
print("Shape of Dataset:")
print(data.shape)

print("\nColumn Names:")
print(data.columns)

print("\nMissing Values:")
print(data.isnull().sum())

# ==========================
# 3. Convert Text Columns to Numbers
# ==========================
data = pd.get_dummies(data, drop_first=True)

print("\nDataset after Encoding:")
print(data.head())

# ==========================
# 4. Features and Target
# ==========================
X = data[[
    "reading score",
    "writing score",
    "gender_male",
    "lunch_standard",
    "test preparation course_none"
]]

y = data["math score"]

print("\nFeature Shape:", X.shape)
print("Target Shape:", y.shape)

# ==========================
# 5. Train-Test Split
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# ==========================
# 6. Create Model
# ==========================
model = LinearRegression()

# ==========================
# 7. Train Model
# ==========================
model.fit(X_train, y_train)

print("\nModel Trained Successfully!")

# ==========================
# 8. Predictions
# ==========================
predictions = model.predict(X_test)

print("\nFirst 5 Predicted Math Scores:")
print(predictions[:5])

print("\nFirst 5 Actual Math Scores:")
print(y_test.head().values)

# ==========================
# 9. Model Evaluation
# ==========================
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print("\n===== Model Performance =====")
print("Mean Absolute Error :", round(mae, 2))
print("Mean Squared Error  :", round(mse, 2))
print("Root Mean Squared Error :", round(rmse, 2))
print("R2 Score :", round(r2, 2))

print("\n===== Linear Regression Performance =====")
print("R2 Score :", round(r2, 2))

# ==========================
# Decision Tree Model
# ==========================
from sklearn.tree import DecisionTreeRegressor

dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)

dt_predictions = dt_model.predict(X_test)

dt_r2 = r2_score(y_test, dt_predictions)

print("\nDecision Tree R2 Score:", round(dt_r2, 2))


# ==========================
# Random Forest Model
# ==========================
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

rf_r2 = r2_score(y_test, rf_predictions)

print("Random Forest R2 Score:", round(rf_r2, 2))

import joblib

# Save the best model
joblib.dump(model, "Model/student_performance_model.pkl")

print("\nBest model saved successfully!")