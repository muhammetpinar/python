import numpy as np
from flask import Flask, request, jsonify
import pickle
import json


# creating a Flask app
app = Flask(__name__)

# Load the model
# model = pickle.load(open('../Models/KNN_model.pkl','rb'))
model = pickle.load(open('C:/Users/muham/Desktop/100+Projects/Build-ML-Models-As-Rest-API-main/Flask/Models/KNN_model.pkl','rb'))

data = request.get_json(force=True)    
varList = []
for val in data.values():
    varList.append(val)


@app.route('/predict', methods = ['POST'])
def pred():

    # Get the data from the POST request.
    data = request.get_json(force=True)    
    varList = []
    for val in data.values():
	    varList.append(val)

    # Make prediction from the saved model
    prediction = model.predict([varList])
    
    # Extract the value
    output = prediction[0]

    #return the output in the json format
    return jsonify(output)


  
# driver function
if __name__ == '__main__':  
    app.run(debug = False, port = 81)