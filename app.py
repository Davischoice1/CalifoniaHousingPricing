import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np 
import pandas as pd

app = Flask(__name__)

# Load the model
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
# Load the scaler
scaling = pickle.load(open('scaling.pkl', 'rb'))  # Ensure 'scalar.pkl' exists

@app.route("/")
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        data = request.json['data']  # Expecting JSON payload with 'data' key
        print("Received data:", data)
        
        # Convert input data to numpy array and reshape
        new_data = scaling.transform(np.array(list(data.values())).reshape(1, -1))
        
        # Predict using the model
        output = regmodel.predict(new_data)
        print("Prediction:", output[0])
        
        # Return the prediction as JSON
        return jsonify({'prediction': float(output[0])})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)


