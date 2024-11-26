import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the trained model
model_file_path = 'inventory_demand_model.pkl'  # Adjust path if necessary
model = joblib.load(model_file_path)

# Load the dataset
file_path = 'Annual_Stock_Summary.csv'  # Your dataset file
data = pd.read_csv(file_path)

# Preprocess data (same steps as during training)
data['Inwards Qty'] = data['Inwards Qty'].fillna(0)
data['Outwards Qty'] = data['Outwards Qty'].fillna(0)
data['Net Movement'] = data['Inwards Qty'] - data['Outwards Qty']
data['Stock Turnover Rate'] = data['Outwards Qty'] / (data['Inwards Qty'] + 1e-9)

# Prepare model data (similar to what you did for training)
model_data = data[['Name ', 'Outwards Qty', 'Inwards Qty']].dropna()

X_new = model_data[['Inwards Qty']]  # Features (same feature used in training)
y_new = model_data['Outwards Qty']  # True values (target variable)

# Make predictions using the trained model
predictions = model.predict(X_new)

# Evaluate model performance using Mean Squared Error (MSE)
mse = mean_squared_error(y_new, predictions)
print(f"Mean Squared Error (MSE) on the new data: {mse}")

# Show the predictions alongside actual values
comparison_df = pd.DataFrame({'Name ': model_data['Name '], 'Actual': y_new, 'Predicted': predictions})
print(comparison_df.head())  # Show first few predictions for analysis

# Plot Actual vs Predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_new, predictions, alpha=0.6)

# Annotate each point with the inventory name
for i, txt in enumerate(model_data['Name ']):
    plt.text(y_new.iloc[i], predictions[i], txt, fontsize=8, alpha=0.7, ha='right', va='bottom')

# Add a red dashed line for perfect prediction (Actual = Predicted)
plt.plot([y_new.min(), y_new.max()], [y_new.min(), y_new.max()], color='red', linestyle='--')

plt.xlabel("Actual Outwards Qty")
plt.ylabel("Predicted Outwards Qty")
plt.title("Actual vs Predicted Outwards Quantity")
plt.tight_layout()
plt.show()
