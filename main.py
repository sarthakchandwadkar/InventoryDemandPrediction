import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from azure.storage.blob import BlobServiceClient
import pyodbc
from azureml.core import Workspace
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from azureml.core import Model



file_path = "Annual_Stock_Summary.csv"
data = pd.read_csv(file_path)
#print(data.head())
#print(data.info())

data['Inwards Qty'] = data['Inwards Qty'].fillna(0)
data['Outwards Qty'] = data['Outwards Qty'].fillna(0)
#print(data.head())

data['Net Movement'] = data['Inwards Qty'] - data['Outwards Qty']
data['Stock Turnover Rate'] = data['Outwards Qty'] / (data['Inwards Qty'] + 1e-9)
#print(data.head())

top_performers = data.nlargest(10, 'Outwards Qty')
bottom_performers = data.nsmallest(10, 'Outwards Qty')

print("Top Performers:")
print(top_performers[['Name ', 'Outwards Qty']])
print("Bottom Performers:")
print(bottom_performers[['Name ', 'Outwards Qty']])
#print(data.head())
#print(data.columns)

plt.figure(figsize=(12, 6))
sns.barplot(x='Name ', y='Outwards Qty', data=top_performers)
plt.title('Top Performing Items')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# AZURE_CONNECTION_STRING =
# blob_service_client =
# container_name = "stockcont"
# blob_client = blob_service_client.get_blob_client(container=container_name, blob="Annual_Stock_Summary.csv")
# processed_file_path = "Annual_Stock_Summary.csv"
# data.to_csv(processed_file_path, index=False)
# with open(processed_file_path, "rb") as file:
#      blob_client.upload_blob(file, overwrite=True)


# conn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};'
#                        'SERVER=serverforsqldb.database.windows.net;'
#                        'DATABASE=StockDB;')
#
# cursor = conn.cursor()
# for index, row in data.iterrows():
#      cursor.execute("INSERT INTO InventoryPerformance (Sr_NO, Name, Inwards_Qty, Outwards_Qty, Net_Movement, Stock_Turnover_Rate) \
#                      VALUES (?, ?, ?, ?, ?, ?)",
#                     row['Sr.NO'], row['Name'], row['Inwards Qty'], row['Outwards Qty'], row['Net Movement'], row['Stock Turnover Rate'])
# conn.commit()
# conn.close()

#Setting up azure ml core workspace

#ws = Workspace.from_config()  # This requires a config.json file from your Azure ML Workspace

model_data = data[['Name ', 'Outwards Qty', 'Inwards Qty']]

model_data = model_data.dropna()

model_data.to_csv("inventory_training_data.csv", index=False)


X = model_data[['Inwards Qty']]  # Features (e.g., inbound quantity or other features)
y = model_data['Outwards Qty']  # Target variable (outward quantity)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")

print("Not saved")
import joblib
joblib.dump(model, 'inventory_demand_model.pkl')

print("Saved")


#Register the model in Azure ML
#model = Model.register(model_path="inventory_demand_model.pkl",model_name="InventoryDemandModel", workspace=ws)

print("Model Uploaded")


