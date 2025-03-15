from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

# Load the trained model
model = joblib.load(r'C:\Crpyto Price\Crypto_Predictive_Model\model\bitcoin_price_predictor.pkl')

# Initialize Flask app
app = Flask(__name__)

# Define the home route
@app.route('/')
def home():
    return render_template(r'C:\Crpyto Price\Crypto_Predictive_Model\templates\index.html')  # Serve the HTML form

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    input_data = request.form.to_dict()

    # Preprocess the input (convert to numpy array and reshape)
    input_array = np.array([float(input_data['Open']), float(input_data['High']), float(input_data['Low']), float(input_data['Volume']), float(input_data['confidence'])])
    input_array = input_array.reshape(1, -1)

    # Make prediction
    prediction = model.predict(input_array)

    # Return the prediction
    return render_template(r'C:\Crpyto Price\Crypto_Predictive_Model\templates\index.html', prediction_text=f'Predicted Bitcoin Close Price: ${prediction[0]:.2f}')

# Run the app
if __name__ == '__main__':
    app.run(debug=True)