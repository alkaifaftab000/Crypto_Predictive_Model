from flask import Flask, request, render_template
import joblib
import numpy as np

# Load the trained model
try:
    model = joblib.load(r'C:\Crpyto Price\Crypto_Predictive_Model\model\bitcoin_price_predictor.pkl')
except FileNotFoundError:
    raise FileNotFoundError("Model file not found. Please check the file path.")

# Initialize Flask app
app = Flask(__name__)

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')  # Serve the HTML form

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    input_data = request.form.to_dict()

    # Preprocess the input (convert to numpy array and reshape)
    try:
        input_array = np.array([
            float(input_data['Open']),
            float(input_data['High']),
            float(input_data['Low']),
            float(input_data['Volume']),
            float(input_data['confidence']),
            float(input_data['Prev_Close']),
            float(input_data['MA_7'])
        ]).reshape(1, -1)
    except (ValueError, KeyError):
        return render_template('index.html', error_text="Invalid input. Please enter valid numeric values.")

    # Make prediction
    prediction = model.predict(input_array)

    # Return the prediction
    return render_template('index.html', prediction_text=f'Predicted Bitcoin Close Price: ${prediction[0]:.2f}')

# Run the app
if __name__ == '__main__':
    app.run(debug=True)