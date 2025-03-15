from flask import Flask, request, render_template
import google.generativeai as genai
import joblib
import os

# Initialize Flask app
app = Flask(__name__)

# Configure Gemini API
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')  # Get API key from environment variable
genai.configure(api_key=GOOGLE_API_KEY)

# Load the Gemini model
model = genai.GenerativeModel('gemini-1.5-flash-latest')
our_model = joblib.load('model/bitcoin_price_predictor.pkl')  # Use relative path


# Define the home route
@app.route('/')
def home():
    return render_template('index.html')  # Serve the HTML form

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # --- Model Prediction (Commented Out) ---
        # prediction = model.predict(input_array)
        # model_prediction = f'Model Predicted Bitcoin Close Price: ${prediction[0]:.2f}'
    try:
        # Get user input from the form
        input_data = request.form.to_dict()

        # Prepare the prompt for Gemini
        prompt = f"""
        Based on the following Bitcoin data, predict the Close price In doller do not print text for the next day :
        - Open: {input_data['Open']}
        - High: {input_data['High']}
        - Low: {input_data['Low']}
        - Volume: {input_data['Volume']}
        - Sentiment Confidence: {input_data['confidence']}
        - Previous Close: {input_data['Prev_Close']}
        - 7-Day Moving Average: {input_data['MA_7']}
        """

        # Generate content using Gemini
        response = model.generate_content(prompt)

        # Return the prediction
        return render_template('index.html', prediction_text=f'$ {response.text}')

    except Exception as e:
        return render_template('index.html', prediction_text=f'An error occurred: {str(e)}')

# Run the app
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)