import pickle
import numpy as np

# Load the stacked_model
with open("stacked_model.pkl", "rb") as f:
    model = pickle.load(f)

# Sample Features
input = np.array([[40,1,66120.267939,8,0,30.568601,0,5]])

# Predict
prediction = model.predict(input)
print("Prediction:", prediction)