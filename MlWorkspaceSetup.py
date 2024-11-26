import json
import joblib
import numpy as np
from azureml.core.model import Model
from sklearn.linear_model import LinearRegression


# Initialize the model
def init():
    # Load the model
    global model
    model_path = Model.get_model_path('Inventory_Demand_Model.pkl')  # Use the name of your registered model
    model = joblib.load(model_path)


# Run inference (prediction)
def run(data):
    # Convert the input data into a numpy array (you may need to adjust based on input data format)
    input_data = np.array(data).reshape(1, -1)

    # Make a prediction using the model
    prediction = model.predict(input_data)

    # Return the prediction as a JSON response
    return json.dumps(prediction.tolist())
