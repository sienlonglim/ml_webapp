from flask import Flask, render_template, request, flash
from modules.MeanEncoder import MeanEncoder
from datetime import datetime
from modules.utils import *
from etl import web_distance_to
import pandas as pd
import numpy as np
import joblib
import yaml
import os


# Flask Routing methods
@app.route("/")
def index():
    return render_template('index.html')

@app.route('/dataset')
def dataset():
    return render_template('dataset.html')

@app.route('/model')
def model():
    return render_template('model.html')


@app.route('/predict', methods=('GET', 'POST'))
@timeit
def predict():
    prediction=None
    if request.method == 'POST':
        data = {'floor_area_sqm': float(request.form['floor_area_sqm']),
                'remaining_lease':float(request.form['remaining_lease']),
                'avg_storey': float(request.form['avg_storey']),}
        
        for_mean_encoding = pd.DataFrame({'town': request.form['town'],
                                          'rooms':float(request.form['rooms'])},
                                          index=[0])

        # Load the model, scaler and encoders           
        model = joblib.load(f'models/gbc_{model_version}.joblib') 
        scaler = joblib.load(f'models/scaler_{model_version}.joblib') 
        # mean_encoder = joblib.load('models/mean_encoder.joblib')
        # Alternative to pickling my own Class, set the encoder using a json
        mean_encoder = MeanEncoder()
        mean_encoder.set_from_json(f'models/encoding_dict_{model_version}.json') 

        df = pd.DataFrame(data, index=[0])

        # Calculate distance to marina bay through OneMap API call
        try:
            df['dist_to_marina_bay'] = web_distance_to(request.form['address'], 'Marina Bay', verbose=0)
        except Exception as error:
            logger.error(error, exc_info=True) 
            flash('Unable to get location of address given, please try again.')
            return render_template('predict.html') 
        
        # Mean encoding
        df['mean_encoded'] = mean_encoder.transform(for_mean_encoding)
        logger.info(f'Prediction for\n{df}')
        df = scaler.transform(df)

        # Prediction
        try:
            prediction = int(model.predict(df)[0])
            logger.info(f'Prediction made at {datetime.now()}: {prediction}')
        except ValueError as error:
            logger.error(error, exc_info=True) 
            flash('No such type of flat found in Town specified, please try again.')
            return render_template('predict.html') 

        
        
        return render_template('predict.html', prediction=prediction)
    
    elif request.method == 'GET':
        return render_template('predict.html', prediction=prediction)


# Main()
if __name__ == "__main__":
    app = Flask(__name__)
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

        # Debug mode should be off if hosted on an external website
        debug= config['local'] 
        
        # Model version is determined by the config file, however if use_curr_datetime is set to True, then it will try to search for most recent model_version
        model_version = config['model_version']
                    
        # Accounts for filepathing local and in pythonanywhere
        if config['local']:
            pass
        else:
            os.chdir(config['web_directory'])
            
    # Getting the credentials for the session and database access
    app.secret_key = os.environ['FLASK_KEY']
    app.run(debug=debug)