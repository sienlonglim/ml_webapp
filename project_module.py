import requests
import numpy as np
import pandas as pd
import json
import time
from requests.exceptions import HTTPError
from pprint import pprint
from functools import wraps
from geopy.distance import geodesic as GD

# Wrapper for timing function calls:
def timeit(func):
    '''
    Wrapper to time function call
    '''
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        '''
        *args and **kwargs here allow parameters for the original function to be taken in
        and passed to the function contained in the wrapper.
        '''
        current_time = time.strftime("%H:%M:%S", time.localtime())
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        time_taken = end-start
        print(f'{func.__name__}() called at \t{current_time} \texecution time: {time_taken:.4f} seconds')
        logging.info(f'{func.__name__}() called at \texecution time: {time_taken:.4f} seconds')
        return result
    return timeit_wrapper

def error_handler(func, max_attempts=3, delay=120):
    '''
    Wrapper to catch and handle errors
    '''
    @wraps(func)
    def error_handler_wrapper(*args, **kwargs):
        '''
        *args and **kwargs here allow parameters for the original function to be taken in
        and passed to the function contained in the wrapper, without needed to declare them in the wrapper function.
        '''
        for i in range(max_attempts):
            try:
                result = func(*args, **kwargs)
            except HTTPError as err:
                logging.error(f'{func.__name__}() encountered {err}')
                # Raise exception if we reach max tries
                if i == max_attempts:
                    raise HTTPError(f'Exceeded max tries of {max_attempts}')
                print(f'{func.__name__}() encountered {err}')

                # err.response gives us the Response object from requests module, we can call .status_code to get the code as int
                if err.response.status_code == 429:
                    print(f'Sleeping for {delay} seconds', end = '\t')
                    time.sleep(delay)
                    print('Retrying...', end='\t')
            except Exception as err:
                logging.error(f'{func.__name__}() encountered {err}') 
                print(f'{func.__name__}() encountered {err}')
                break
            else:
                return result
    return error_handler_wrapper

@timeit
@error_handler
def get_token(location: str):
    '''
    Function to check if API token is still valid and updates API token if outdated
    ##Parameters
        location: filepath (str)
    Returns API token : str
    '''
    with open(location, 'r+') as fp:
        file = fp.read()
        data = json.loads(file)
        response = requests.post("https://developers.onemap.sg/privateapi/auth/post/getToken", data=data)
        token = response.json()
        if token['access_token'] != data['access_token']:
            print(f"New token found")
            data['access_token'] = token['access_token']
            data['expiry_timestamp'] = token['expiry_timestamp']
            fp.seek(0)
            json.dump(data, fp = fp, indent=4)
            print('Updated token json')
            data = json.loads(file)
        return data['access_token']

@timeit
@error_handler
def datagovsg_api_call(url: str, sort: str = 'month desc', limit: int = 100, 
                       months:list =[1,2,3,4,5,6,7,8,9,10,11,12], 
                       years:list =["2022"]) -> pd.DataFrame:
    '''
    Function to build the API call and construct the pandas dataframe
    ## Parameters
    url: str
        url for API, with resource_id parameters
    sort: str
        field, by ascending/desc, default by Latest month
    limit: int
        maximum entries (API default by OneMap is 100, if not specified)
    months: list
        months desired, int between 1-12
    years: list
        months desired , int
    Returns Dataframe of data : pd.DataFrame
    '''
    month_dict = '{"month":['
    for year in years:
        for month in months: # months 1-12
            month_dict = month_dict + f'"{year}-{str(month).zfill(2)}", '
    month_dict = month_dict[:-2] # Cancel out extra strings <, >
    month_dict = month_dict + ']}'
    url = url+f'&sort={sort}&filters={month_dict}'
    if limit: # API call's default is 100 even without specifying
        print(f'Call limit : {limit}')
        url = url+f'&limit={limit}'
    pprint(f'API call = {url}')
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    df = pd.DataFrame(data['result']['records'])
    return df

