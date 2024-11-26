import json
import joblib
import numpy as np
from azureml.core.model import Model
from sklearn.linear_model import LinearRegression


# Initialize the model
def init():
    global model
    try:
        # Load the model from Azure ML's registered model store
        model_path = Model.get_model_path('InventoryDemandModel')  # Use the name of your registered model
        model = joblib.load(model_path)
    except Exception as e:
        # If there's an error loading the model, log the error and raise an exception
        raise Exception(f"Error loading the model: {str(e)}")


# Run inference (prediction)
def run(data):
    try:
        # Convert the input data into a numpy array (you may need to adjust based on input data format)
        input_data = np.array(data).reshape(1, -1)

        # Ensure that the input data has the correct shape for prediction
        if input_data.shape[1] != model.coef_.shape[0]:
            raise ValueError("Input data has an incorrect number of features.")

        # Make a prediction using the model
        prediction = model.predict(input_data)

        # Return the prediction as a JSON response
        return json.dumps(prediction.tolist())

    except Exception as e:
        # If there's an error during prediction, return an error message
        return json.dumps({"error": f"Error during prediction: {str(e)}"})


