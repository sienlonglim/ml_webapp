from flask import Flask, render_template, request, url_for, flash, redirect
from markupsafe import escape
from pprint import pprint
from functools import wraps
from static.MeanEncoder import MeanEncoder
from geopy.distance import geodesic as GD
import pandas as pd
import numpy as np
import joblib
import json
import time
import logging
import requests

app = Flask(__name__)
debug= True # Debug mode should be off if hosted on an external website

# Customs classes, functions, decorators
def get_value_from_json(json_file, key, sub_key=None):
   '''
   Function to read json config files
   ## Parameters
    json_file : str, pathname to json file
    key : str, key
    sub_key : nested key, if applicable
   '''
   with open(json_file) as f:
    data = json.load(f)
    if sub_key:
        return data[key][sub_key]
    else:
        return data[key]

def timeit(func):
    '''
    Decorator to time function call
    '''
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        time_taken = end-start
        logging.info(f'{func.__name__}() called at \texecution time: {time_taken:.4f} seconds')
        return result
    return timeit_wrapper

def error_handler(func):
    '''
    Decorator to catch and handle errors
    '''
    @wraps(func)
    def error_handler_wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
        except Exception as err:
            logging.error(f'{func.__name__}() encountered {err}') 
        else:
            return result
    return error_handler_wrapper

@timeit
def distance_to(from_address : str, to_address : str, verbose : int=0):
    '''
    Function to determine distance to a location
    ## Parameters
    postcode : int containing postcode
    to_address : str
        place and streetname
    verbose : int
        whether to show the workings of the function

    Returns np.Series of distance between input and location
    '''
    if not isinstance(from_address, str) or not isinstance(to_address, str):
        raise ValueError('Input must be string')
    
    # get from address
    call = f'https://developers.onemap.sg/commonapi/search?searchVal={from_address}&returnGeom=Y&getAddrDetails=Y'
    response = requests.get(call)
    response.raise_for_status()
    data = response.json()
    from_coordinates = (float(data['results'][0]['LATITUDE']), float(data['results'][0]['LONGITUDE']))
    if verbose==1:
        print(f'Coordinates of {from_address} : {from_coordinates}')

    # get to address
    call = f'https://developers.onemap.sg/commonapi/search?searchVal={to_address}&returnGeom=Y&getAddrDetails=Y'
    response = requests.get(call)
    response.raise_for_status()
    data = response.json()
    to_coordinates = (float(data['results'][0]['LATITUDE']), float(data['results'][0]['LONGITUDE']))
    if verbose==1:
        print(f'Coordinates of {to_address} : {to_coordinates}')

    # calculate geodesic distance
    geodesic_dist = GD(from_coordinates, to_coordinates).kilometers
    return np.round(geodesic_dist,2)

# Getting the credentials for the session and database access
# Modify route according to directory
app.secret_key = get_value_from_json("venv/secrets.json", "flask", "SECRET_KEY")

# Flask Routing methods
@app.route("/")
def index():
    '''
    Index page
    '''
    return render_template('index.html')

@app.route('/dataset')
def dataset():
    return render_template('dataset.html')

@app.route('/model')
def model():
    return render_template('model.html')

@app.route('/predict', methods=('GET', 'POST'))
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
        # Modify route according to directory
        model = joblib.load('models/gbc_2023_01_to_04.joblib')
        scaler = joblib.load('models/scaler.joblib')
        mean_encoder = joblib.load('models/mean_encoder.joblib')
        df = pd.DataFrame(data, index=[0])

        # Calculate distance to marina bay through OneMap API call
        try:
            df['dist_to_marina_bay'] = distance_to(request.form['address'], 'Marina Bay', verbose=0)
        except Exception as error:
            logging.error(error) 
            flash('Unable to get location of address given, please try again.')
            return render_template('predict.html') 
        
        # Mean encoding
        df['mean_encoded'] = mean_encoder.transform(for_mean_encoding)
        df = scaler.transform(df)

        # Prediction
        try:
            prediction = np.round(model.predict(df)[0])
        except ValueError as error:
            logging.error(error) 
            flash('No such type of flat found in Town specified, please try again.')
            return render_template('predict.html') 

        return render_template('predict.html', prediction=prediction)
    
    elif request.method == 'GET':
        return render_template('predict.html', prediction=prediction)


# Main()
if __name__ == "__main__":
    logging.basicConfig(filename='app.log', filemode='a', level=logging.ERROR, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    app.run(debug=debug)