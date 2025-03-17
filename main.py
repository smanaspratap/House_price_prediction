import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from numpy import sqrt
import matplotlib.pyplot as plt

# Step 1: Load the dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame

# Step 2: Data Preprocessing
scaler = StandardScaler()
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 3: Model Selection and Training
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
}

# Step 4 & 5: Evaluation and Visualization
results = {}
plt.figure(figsize=(12, 6))

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = sqrt(mean_squared_error(y_test, y_pred))
    results[name] = {"MAE": mae, "RMSE": rmse}

    # Visualization
    plt.scatter(y_test, y_pred, alpha=0.5, label=name)

# Best-fit line for reference
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
plt.title('Predicted vs Actual House Prices - Model Comparison')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.legend()
plt.grid(True)
plt.show()

# Display Results
for model, metrics in results.items():
    print(f"{model} -> MAE: {metrics['MAE']:.4f}, RMSE: {metrics['RMSE']:.4f}")
