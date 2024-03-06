'''
This script build the model using the train and test data
'''
from modules.MeanEncoder import MeanEncoder
from modules.utils import *
import os
import yaml
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import  GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

if __name__ ==  '__main__':
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
        model_version = config['save_model_version']
        train = config["train"]
        test = config["test"]

        # if config['automation'] & datetime.now().day != 30:
        #     model_logger.info('Exiting Model Building script - script will only run on 30th of each month')
        #     sys.exit()

        # Accounts for filepathing local and in pythonanywhere
        if config['local']:
            pass
        else:
            os.chdir(config['web_directory'])
    
    # Get the correct etl_logger
    model_logger = logging.getLogger('model')
    model_logger.info(f"{'-'*50}New Model building run started {'-'*50}")
    model_logger.info(f'Data settings:')
    model_logger.info(f'\ttrain: {train}')
    model_logger.info(f'\ttest: {test}')
    model_logger.info(f'\tmodel version: {model_version}')

    # Loading training data
    model_logger.info(f'Loading dataframe and generating features...')
    df = pd.read_csv(train, index_col=0)
    df.dropna(inplace=True)

    # Feature creation and selection
    numerical_columns = df[['resale_price', 'floor_area_sqm', 'remaining_lease', 'rooms', 'avg_storey', 'dist_to_marina_bay', 'dist_to_station_0']]
    mean_encoder = MeanEncoder(measure='mean')
    mean_encoder.fit(df, columns=['town', 'rooms'], target_column='resale_price')
    town_mean_price = mean_encoder.transform(df)
    train_df = pd.concat([numerical_columns, town_mean_price], axis =1)

    X_unscaled = train_df[['floor_area_sqm', 'remaining_lease', 'avg_storey', 'dist_to_marina_bay', 'mean_encoded']] 

    scaler = MinMaxScaler()
    y = train_df.iloc[:,0]
    X = scaler.fit_transform(X_unscaled)

    # Hyperparameter tuning
    model_logger.info('Starting hyperparameter tuning...')
    param_distributions = {'max_depth' : [3,5],
                        'n_estimators' : [50,100,150],
                        'learning_rate' : [0.01,0.1,1],
                        'max_features': [None, 'sqrt', 'log2']
                        }

    random_cv = RandomizedSearchCV(estimator=GradientBoostingRegressor(random_state=42),
                                scoring= 'r2', 
                                param_distributions= param_distributions, 
                                n_iter= 15,
                                cv= 5, 
                                verbose= 1,
                                n_jobs=2)

    random_cv.fit(X, y)
    model_logger.info(f'\tBest Parameters: {random_cv.best_params_}')
    model_logger.info(f'\t- Validation R2 score: {np.round(random_cv.best_score_,3)}')
    best_gbc = random_cv.best_estimator_

    # Test against latest data
    test_df = pd.read_csv(test, index_col=0)
    mean_encoded = mean_encoder.transform(test_df)
    test_df = pd.concat([test_df, mean_encoded], axis =1)
    test_df = test_df[['resale_price','floor_area_sqm', 'remaining_lease', 'avg_storey', 'dist_to_marina_bay', 'mean_encoded']]
    X_test = test_df.iloc[:,1:]
    y_test = test_df.iloc[:,0]
    X_test = scaler.transform(X_test)
    y_pred = best_gbc.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    model_logger.info(f'\t- Reported R2 score = {np.round(r2,3)}')
    model_logger.info(f'\t- Reported MAE = {int(mae)}')


    # Saving
    model_logger.info('\nSaving...')
    model_logger.info(f"\tScaler object saved as {joblib.dump(scaler, f'models/scaler_{model_version}.joblib')}")
    model_logger.info(f"\tMean encoder object as{joblib.dump(mean_encoder, f'models/mean_encoder_{model_version}.joblib')}")
    model_logger.info(f"\tMean encoding Json exported as {mean_encoder.export_to_json(f'models/encoding_dict_{model_version}.json')}")
    model_logger.info(f"\tML model saved as {joblib.dump(best_gbc, f'models/gbc_{model_version}.joblib')}")
    model_logger.info(f'\nAll jobs completed @ {datetime.now()}')