# import numpy as np
# import joblib
# import sklearn
#
# print("Numpy version:", np.__version__)
# print("Scikit-learn version:", sklearn.__version__)

import numpy as np
import joblib
import sklearn

# Load your model
model = joblib.load('inventory_demand_model.pkl')

# Simulate input data
X_test = np.array([[1]])

# Make a prediction
prediction = model.predict(X_test)

print("Prediction:", prediction)




