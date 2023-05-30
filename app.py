from flask import Flask, render_template, request, url_for, flash, redirect
from markupsafe import escape
from pprint import pprint
from functools import wraps
import pandas as pd
import joblib
import json
import time
import logging

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
        current_time = time.strftime("%H:%M:%S", time.localtime())
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        time_taken = end-start
        print(f'{func.__name__}() called at \t{current_time} \texecution time: {time_taken:.4f} seconds')
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
            print(f'{func.__name__}() encountered {err}')
            logging.error(f'{func.__name__}() encountered {err}') 
        else:
            return result
    return error_handler_wrapper

# Getting the credentials for the session and database access
app.secret_key = get_value_from_json("venv/secrets.json", "flask", "SECRET_KEY")
config = get_value_from_json("venv/secrets.json", "mysql_connector")

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
    if request.method == 'POST':
        data = {'town': request.form['town'],
                'floor_area_sqm': float(request.form['floor_area_sqm']),
                'avg_storey': float(request.form['avg_storey']),
                'rooms':float(request.form['rooms']),
                'remaining_lease':float(request.form['remaining_lease']),
        }
        model = joblib.load('models/gbc_2023_01_to_04.joblib')
        scaler = joblib.load('models/scaler.joblib')
        mean_encoder = joblib.load('models/mean_encoder.joblib')
        df = pd.DataFrame(data, index=[0])
        df = scaler.transform(df)
        df = mean_encoder.transform(df)
        pprint(df)
        
        return render_template('predict.html')
    elif request.method == 'GET':
        return render_template('predict.html')



if __name__ == "__main__":
    logging.basicConfig(filename='app.log', filemode='a', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    app.run(debug=debug)