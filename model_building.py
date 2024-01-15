'''
This script build the model using the train and test data
'''
import sys
import os
import yaml
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import  GradientBoostingRegressor
import joblib
from modules import MeanEncoder

if __name__ ==  '__main__':
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

        if config['automation'] & datetime.now().day != 30:
            print('Exiting Model Building script - script will only run on 30th of each month')
            sys.exit()

        # Accounts for filepathing local and in pythonanywhere
        if config['local']:
            pass
        else:
            os.chdir(config['web_prefix'])
    
    # Loading training data
    print(f'Loading dataframe and generating features...')
    df = pd.read_csv(config['train'], index_col=0)
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
    print('Starting hyperparameter tuning...')
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
    print('Best Parameters', random_cv.best_params_)
    print('Best R2 score', np.round(random_cv.best_score_,3))
    best_gbc = random_cv.best_estimator_

    # Test against latest data
    test_df = config['test']

    # Saving
    model_version = config['save_model_version']
    print('\nSaving...')
    print(f"Scaler object saved as {joblib.dump(scaler, f'models/scaler_{model_version}.joblib')}")
    print(f"Mean encoder object as{joblib.dump(mean_encoder, f'models/mean_encoder_{model_version}.joblib')}")
    print(f"Mean encoding Json exported as {mean_encoder.export_to_json(f'models/encoding_dict_{model_version}.json')}")
    print(f"ML model saved as {joblib.dump(best_gbc, f'models/gbc_{model_version}.joblib')}")
    print(f'\nAll jobs completed @ {datetime.now()}')