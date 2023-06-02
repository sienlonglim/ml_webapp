import pandas as pd

class MeanEncoder():
    '''
    Custom class encoder to deal with mean/median encoding
    '''
    def __init__(self, measure:str='mean'):
        self.encoder_dict_ = None
        self.columns_ = None
        self.measure_ = measure
        self.target_column_ = None

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

    def export_to_json(self, filepath):
        '''
        Export the underlying variables to a json file
            The dictionary with tuples is written as a str first, to be read later using eval()
        Returns a json file to the specified filepath
        '''
        import json
        export_dict = {'encoder_dict': str(self.encoder_dict_),
                        'columns': self.columns_,
                        'target_column': self.target_column_}
        with open(filepath, 'w')as f:
            json.dump(export_dict, f, indent=4)
