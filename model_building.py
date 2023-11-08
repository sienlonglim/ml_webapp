'''
This script build the model using the train and test data
'''
import sys
import numpy as np
import pandas as pd
import yaml
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import  GradientBoostingRegressor
import joblib

class MeanEncoder():
    '''
    Custom class encoder to deal with mean/median encoding
    '''
    def __init__(self, measure:str='mean'):
        self.encoder_dict_ = None
        self.columns_ = None
        self.measure_ = measure
        self.target_column_ = None
        self.filepath = 'No filepath specified'
    
    def __str__(self):
        return self.encoder_dict_

    def fit(self, X : pd.DataFrame, columns : list, target_column : str)->None:
        '''
        Fit to dataframe to create encoder_dict_ (dictionary) for data mapping
        ## Parameters
            X : pd.DataFrame object
            columns : list of strings, indicating columns to groupby
            target_column : str, desired output (must be numeric)
        Returns None
        '''
        self.columns_ = columns
        self.target_column_ = target_column
        if self.measure_ == 'mean':
            self.encoder_dict_ = X.groupby(self.columns_)[self.target_column_].mean(numeric_only=True).to_dict()
        elif self.measure_ == 'median':
            self.encoder_dict_ = X.groupby(self.columns_)[self.target_column_].median(numeric_only=True).to_dict()
    
    def transform(self, X : pd.DataFrame)->pd.Series:
        '''
        Transform dataframe by mapping data using encoder_dict_
        ## Parameters
            X : pd.DataFrame object
        Returns pd.Series of encoded data
        '''
        def columns_to_tuple(df, columns):
            '''
            Function to combined columns as a tuple for dictionary mapping
            '''
            temp = []
            for column in columns:
                temp.append(df[column])
            return tuple(temp)
        
        row_tuple = X.apply(columns_to_tuple, columns = self.columns_, axis=1)
        row_tuple.name = f'{self.measure_}_encoded'
        output =  row_tuple.map(self.encoder_dict_)
        return output
    
    def set_from_json(self, filepath):
        '''
        Manually set an encoding dictionary
        '''
        import json
        with open(filepath) as f:
            data = json.load(f)
            self.encoder_dict_ = data['encoder_dict']
            # Note eval() is used to read the str to get back the tuple
            self.encoder_dict_ = eval(self.encoder_dict_)
            self.columns_ = data['columns']
            self.target_column_ = data['target_column']
        return filepath

    def export_to_json(self, filepath):
        '''
        Export the underlying variables to a json file
            The dictionary with tuples is written as a str first, to be read later using eval()
        Returns a json file to the specified filepath
        '''
        import json
        self.filepath = filepath
        export_dict = {'encoder_dict': str(self.encoder_dict_),
                        'columns': self.columns_,
                        'target_column': self.target_column_}
        with open(filepath, 'w')as f:
            json.dump(export_dict, f, indent=4)
        return filepath


if __name__ ==  '__main__':
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

        if config['automation'] & datetime.now().day != 30:
            print('Exiting Model Building script - script will only run on 30th of each month')
            sys.exit()

        # Accounts for filepathing local and in pythonanywhere
        if config['local']:
            filepath_prefix = ''
        else:
            filepath_prefix = config['web_prefix']
    
    # Loading training data
    print(f'Loading dataframe and generating features...')
    df = pd.read_csv(f'{filepath_prefix}static/train.csv', index_col=0)
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

    # Saving
    model_version = config['save_model_version']
    print('\nSaving...')
    print(f"Scaler object saved as {joblib.dump(scaler, f'{filepath_prefix}models/scaler_{model_version}.joblib')}")
    print(f"Mean encoder object as{joblib.dump(mean_encoder, f'{filepath_prefix}models/mean_encoder_{model_version}.joblib')}")
    print(f"Mean encoding Json exported as {mean_encoder.export_to_json(f'{filepath_prefix}models/encoding_dict_{model_version}.json')}")
    print(f"ML model saved as {joblib.dump(best_gbc, f'{filepath_prefix}models/gbc_{model_version}.joblib')}")
    print(f'\nAll jobs completed @ {datetime.now()}')