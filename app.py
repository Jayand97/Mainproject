from flask import Flask, request, render_template
import pickle
import pandas as pd
import warnings
#warnings.filterwarnings('ignore')
import  bz2

app = Flask(__name__)



# Load the trained model and scaler from the pickle file
with bz2.open('model.pbz2', 'rb') as f:
    rf_model = pickle.load(f)

# Load the scaler with compression
with bz2.open('scaler.pbz2', 'rb') as f:
   scaler = pickle.load(f)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the user input features
        type = int(request.form['type'])
        amount = float(request.form['amount'])
        oldbalance = float(request.form['oldb'])
        newbalance = float(request.form['newb'])
       

        # Scale the input features
        input_features = scaler.transform([[type, amount, oldbalance, newbalance]])

        # Make a prediction
        prediction = rf_model.predict(input_features)

        # Print whether a person will purchase the product or not
        if prediction[0] == 1:
            result = 'The Transaction detected as Fraud.'
        else:
            result = 'The Transcation is Not-Fraud.'

        return render_template('result.html',result=result)
   
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)