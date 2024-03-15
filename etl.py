'''
This script performs the ETL process, config.yaml specifies which months' data to extract, transform, and load as a csv
1. API call is made to extract monthly data from data.gov.sg
2. Script performs tranformation of data, and searches for geolocations and routing to city center + nearest mrt stations
3. Splits data into train (all previous months) and test set (most recent month)
'''
import os
import requests
import requests_cache
import numpy as np
import pandas as pd
import json
import yaml
from datetime import datetime
from geopy.distance import geodesic as GD
from modules.utils import *

# Logger here is the debugger logger
logger.info(f"{'-'*50}New run started {'-'*50}")

# Declaring all the functions
@error_handler
def get_token(location: str):
    '''
    Function to check if API token is still valid and updates API token if outdated
    The API token is necessary for routing API calls to determine distance to nearest MRT
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
            logger.info(f"New token found")
            data['access_token'] = token['access_token']
            data['expiry_timestamp'] = token['expiry_timestamp']
            fp.seek(0)
            json.dump(data, fp = fp, indent=4)
            logger.info('Updated token json')
            data = json.loads(file)
        return data['access_token']

@error_handler
def datagovsg_api_call(url: str, sort: str = 'month desc', limit: int = 10000, verbose: bool = False,
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
        maximum entries (API default by datagov is 100, if not specified)
    verbose: bool
        whether to print out the calls
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
    url = url+f'&limit={limit}'
    if verbose:
        logger.info(f'Call limit : {limit}')
        logger.info(f'API call = {url}')
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    df = pd.DataFrame(data['result']['records'])
    return df

@error_handler
def datagovsg_api_call_v2(resource_id: str='d_8b84c4ee58e3cfc0ece0d773c8ca6abc'
                            , verbose: bool = True, **kwargs) -> pd.DataFrame:
    '''
    Function to build the API call and construct the pandas dataframe, 
    data.gov now limits the calls, so it is safer to call month by month to not have data truncated
    ## Parameters
    resource_id: str
        resource_id for API
    verbose: bool
            whether to print out the calls
    **kwargs:
        refer to API guide https://guide.data.gov.sg/developers/api-v2
        e.g.:
            year: str
                year
            month: 
                month desired, int between 1-12
            sort: str
                field, by ascending/desc, default by Latest month
            limit: int
                maximum entries (API default by datagov is 100, if not specified)
            
    Returns Dataframe of data : pd.DataFrame
    '''
    payload = {}
    if kwargs.get("year") and kwargs.get("month"):
        year_month = f'{kwargs.get("year")}-{str(kwargs.get("month")).zfill(2)}'
        month_filter = {"month": [year_month]}
        # To add a nested dictionary for request get, we need to parse the nested dictionary into json format
        payload["filters"] = json.dumps(month_filter)
    payload["limit"] = 10000
    payload["sort"] = 'month desc'

    url = f"https://data.gov.sg/api/action/datastore_search?resource_id={resource_id}"
    response = requests.get(url, params=payload)
    response.raise_for_status()
    if verbose:
        logger.info(f'Response: {response.status_code} \tGet request call: {response.url}')
    data = response.json()
    df = pd.DataFrame(data['result']['records'])
    return df

@timeit
def clean_df(df: pd.DataFrame):
    '''
    Function to clean the raw dataframe
    ##Parameters
    pd.DataFrame
    ##Cleaning done
        1. Reindexed dataframe using _id (unique to every resale transaction)
        2. Changed room types into float values, with Executive as 4.5 rooms (extra study/balcony), and Multigeneration 6 rooms
        3. Storey range was converted to avg_storey, the avg floor would be used (every value is a difference of 3 storeys)
        4. Resale_price, Floor area converted to float values
        5. Month was converted into datetime format, to be used to detrend the time series moving average
        6. Year/Month was separated into Year and Month for visualisation purposes
        7. Remaining lease was converted into remaining months (float)
        8. Update capitalisation and street naming conventions (for purpose of API call later)
        9. Categorised towns into regions (North, West, East, North-East, Central) 
    Returns the cleaned dataframe
    '''
    try:
        # Start
        # Step 1: set index to overall id
        step = 1
        df.set_index('_id', inplace=True)
            
        # Step 2: Create feature "rooms", "avg_storey"
        def categorise_rooms(flat_type):
            '''
            Helper function for categorising number of rooms
            '''
            if flat_type[0] == 'E' or flat_type[0] == 'M':
                return 5.5
            else:
                return float(flat_type[0])
        
        step = 2
        df['rooms'] = df['flat_type'].apply(categorise_rooms)
        step = 3
        df['avg_storey'] = df['storey_range'].apply(lambda x: (int(x[:2])+int(x[-2:]))/2)

        # Step 4-6: Change dtypes
        df['resale_price'] = df['resale_price'].astype('float')
        df['floor_area_sqm'] = df['floor_area_sqm'].astype('float')
        step = 5
        df['year_month'] = df['month']
        df['year_month_dt'] = pd.to_datetime(df['month'], format="%Y-%m")
        step = 6
        df['year'] = df['year_month_dt'].dt.year
        df['month'] = df['year_month_dt'].dt.month
        step = 7
        df['lease_commence_date'] = df['lease_commence_date'].astype('int')
        
        # Calculate remaining_lease
        def year_month_to_year(remaining_lease):
            '''
            Helper function to change year & months, into years (float)
            '''
            remaining_lease = remaining_lease.split(' ')
            if len(remaining_lease) > 2:
                year = float(remaining_lease[0]) + float(remaining_lease[2])/12
            else:
                year = float(remaining_lease[0])
            return year
        
        df['remaining_lease'] = df['remaining_lease'].apply(year_month_to_year)

        step = 8
        # Step 8: Change capitalization of strings
        for column in df.columns:
            if df[column].dtype == 'O':
                df[column] = df[column].str.title()
        
        # Update address abbreviations for onemap API call
        abbreviations = {'Sth':'South', 
                        '[S][t][^.ri]':'Street ', 
                        '[S][t]$':'Street',
                        '[S][t][.]':'Saint', 
                        'Nth':'North', 
                        'Ave':'Avenue', 
                        'Dr':'Drive', 
                        'Rd':'Road'}
        for abbreviation, full in abbreviations.items():
            df['street_name'] = df['street_name'].str.replace(abbreviation, full, regex=True)
        # Concatenate block and street into a full address
        df['address'] = df['block'] + ', ' + df['street_name']
        
        # Step 9: Categorise town regions
        step = 9
        town_regions = {'Sembawang' : 'North',
                    'Woodlands' : 'North',
                    'Yishun' : 'North',
                    'Ang Mo Kio' : 'North-East',
                    'Hougang' : 'North-East',
                    'Punggol' : 'North-East',
                    'Sengkang' : 'North-East',
                    'Serangoon' : 'North-East',
                    'Bedok' : 'East',
                    'Pasir Ris' : 'East',
                    'Tampines' : 'East',
                    'Bukit Batok' : 'West',
                    'Bukit Panjang' : 'West',
                    'Choa Chu Kang' : 'West',
                    'Clementi' : 'West',
                    'Jurong East' : 'West',
                    'Jurong West' : 'West',
                    'Tengah' : 'West',
                    'Bishan' : 'Central',
                    'Bukit Merah' : 'Central',
                    'Bukit Timah' : 'Central',
                    'Central Area' : 'Central',
                    'Geylang' : 'Central',
                    'Kallang/Whampoa' : 'Central',
                    'Marine Parade' : 'Central',
                    'Queenstown' : 'Central',
                    'Toa Payoh' : 'Central'}      
        df['region'] = df['town'].map(town_regions)
    except Exception as err:
        logger.info(f"Error at step {step}, error message: {err}", exc_info=True)
    else:
        # Reorder columns
        
        df = df[['resale_price', 'year', 'month', 'year_month', 'region', 'town', 'rooms', 'avg_storey', 'floor_area_sqm', 'remaining_lease', 'address']]
                # Unused columns - 'lease_commence_date', 'flat_model', 'storey_range', 'flat_type', 'block', 'street_name'
    return df

@error_handler
def get_location_data(address_df: pd.DataFrame, verbose : int=0, cached_session : requests_cache.CachedSession =None):
    '''
    Function to carry out API call for Geodata
    ## Parameters
    address_df : pd.DataFrame
        DataFrame that contains a combination of ['block'] and ['street_name'] as ['address'], and ['town']
    verbose : int
        1 to verbose calls
        2 to verbose results
    '''
    # Getting latitude, longitude, postal code
    def get_lat_long(address_df : pd.DataFrame, sleeptime : float =0.15):
        '''
        The actual API call to be called row-wise to get latitude, longitude, and postal code
        ## Parameters
        address_df : pd.DataFrame
            DataFrame that contains a combination of ['block'] and ['street_name'] as ['address'], and ['town']
        sleeptime : float
            Incorporates sleep time to not exceed a max of 250 calls per min
            Default 0.15s, not required if we are caching call
        '''
        
        # Lag time between calls - No longer needed with Cache, since we will not likely exceed the call limit
        # time.sleep(sleeptime)

        # API call
        try:
            address = address_df['address']
            if 'Jln Batu' in address:
                address = address.replace('Jln Batu', 'JALAN BATU DI TANJONG RHU')
            elif '27, Marine Cres' in address:
                address = address.replace('Marine Cres', 'MARINE CRESCENT MARINE CRESCENT VILLE')
            elif '215, Choa Chu Kang Ctrl' in address:
                address = '680215'
            elif '216, Choa Chu Kang Ctrl' in address:
                address = '680216'
                
            call = f"https://www.onemap.gov.sg/api/common/elastic/search?searchVal={address}&returnGeom=Y&getAddrDetails=Y"
            # call = f'https://developers.onemap.sg/commonapi/search?searchVal={address}&returnGeom=Y&getAddrDetails=Y' # Old API call
            # Caching is enabled in the session
            response = cached_session.get(call)
            response.raise_for_status()
            data = response.json()
            if verbose >0:
                logger.info(f'Response: {response.status_code} \tGet request call: {response.url}')
            if verbose >1:
                logger.info(data)

            # Returns a the results in string
            return data['results'][0]['LATITUDE'] + ',' + data['results'][0]['LONGITUDE'] + ' ' + data['results'][0]['POSTAL']
        
        except Exception as err:
            logger.error(f'Error occurred - get_lat_long() API call: {err} on the following call:', exc_info=True)
            return '0,0 0' # Still return 0 values

    def to_numpy_array(lat_long_df):
        # Build a numpy array from latitude and longitude
        combi = np.array([lat_long_df[0], lat_long_df[1]])
        return combi
    
    # This calls the API call function row wise
    position = address_df.apply(get_lat_long, axis=1)

    try:
        # Split the string into two columns (column 0 is the latitude and longitude, column 1 is the postal code)
        temp_df = position.str.split(expand=True)
        # Postal code
        temp_df.iloc[:,1] = temp_df.iloc[:,1].apply(lambda x: 0 if x=='NIL' else x)
        temp_df.iloc[:,1] = temp_df.iloc[:,1].astype('int')
        # Latitude and longitude split (by ,)
        lat_long_df = temp_df.iloc[:,0].str.split(pat=',', expand=True)
        lat_long_df = lat_long_df.astype('float')
        # Convert into numpy array, for faster matrix operations later
        numpy_array = lat_long_df.apply(to_numpy_array, axis=1)
        
    except Exception as err:
        logger.error(f"Error occurred - Splitting data : {err}")
    else:
        geo_data_df = pd.concat([temp_df, lat_long_df, numpy_array], axis=1)
        geo_data_df.columns = ['lat_long', 'postal_code', 'latitude', 'longitude', 'numpy_array']
        return geo_data_df

@error_handler
def multiple_distance_to(df_series : pd.Series, to_address : str , dist_type : str='latlong', verbose : int=0):
    '''
    Function to determine distance to a location (from a series of locations in a dataframe
    ## Parameters
    df_series : pd.Series contains numpy array containing [latitude, longitude]
    to_address : str
        place and streetname
    dist_type : str
        type of distance (latlong, or geodesic)
    verbose : int
        whether to show the workings of the function

    Returns np.Series of distance between input and location
    '''
    # if an address is given
    if isinstance(to_address, str):
        call = f"https://www.onemap.gov.sg/api/common/elastic/search?searchVal={to_address}&returnGeom=Y&getAddrDetails=Y"
        # call = f'https://developers.onemap.sg/commonapi/search?searchVal={to_address}&returnGeom=Y&getAddrDetails=Y' # Old API Call
        response = requests.get(call)
        response.raise_for_status()
        data = response.json()
        to_coordinates = np.array([float(data['results'][0]['LATITUDE']), float(data['results'][0]['LONGITUDE'])])

    if verbose==1:
        logger.info(f'Coordinates of {to_address} : {to_coordinates}')

    def matrix_operations(from_coordinates, to_coordinates):
        # Matrix substraction to get difference 
        distance_diff = from_coordinates - to_coordinates
        absolute_dist = np.absolute(distance_diff)

        #Matrix sum over latitude and longitude of each entry
        sum_of_distances = np.sum(absolute_dist)

        if verbose==2:
            logger.info(f'Difference in distances: \n{distance_diff}')
            logger.info()
            logger.info(f'Absolute difference: \n{absolute_dist}')
            logger.info()
            logger.info(f'Sum of distances \n {sum_of_distances}')
        
        return sum_of_distances

    def geodesic_operations(from_coordinates, coordinates):
        from_coordinates = tuple(from_coordinates)
        coordinates = tuple(coordinates)
        geodesic_dist = GD(from_coordinates, coordinates).kilometers
        return np.round(geodesic_dist,2)
    
    if dist_type == 'geodesic':
        diff_dist = df_series.apply(geodesic_operations, coordinates=to_coordinates)
    else:
        diff_dist = df_series.apply(matrix_operations, coordinates=to_coordinates)

    return diff_dist

@error_handler
def single_distance_to(from_address : str, to_address : str, verbose : int=0):
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
    call = f"https://www.onemap.gov.sg/api/common/elastic/search?searchVal={from_address}&returnGeom=Y&getAddrDetails=Y"
    response = requests.get(call)
    response.raise_for_status()
    data = response.json()
    from_coordinates = (float(data['results'][0]['LATITUDE']), float(data['results'][0]['LONGITUDE']))
    if verbose==1:
        logger.info(f'Coordinates of {from_address} : {from_coordinates}')

    # get to address
    call = f"https://www.onemap.gov.sg/api/common/elastic/search?searchVal={to_address}&returnGeom=Y&getAddrDetails=Y"
    response = requests.get(call)
    response.raise_for_status()
    data = response.json()
    to_coordinates = (float(data['results'][0]['LATITUDE']), float(data['results'][0]['LONGITUDE']))
    if verbose==1:
        logger.info(f'Coordinates of {to_address} : {to_coordinates}')

    # calculate geodesic distance
    geodesic_dist = GD(from_coordinates, to_coordinates).kilometers
    return np.round(geodesic_dist,2)

@error_handler
def get_mrt_coordinates(mrt_stations: str=None, filepath: str=None)-> None:
    '''
    Function to API call for MRT station coordinates and write to json file
    ## Parameters
    mrt_stations : list
        list of mrt station names, default to All stations if nothing is given
    filepath : str
        filepath and name of json file to write to, should end with .json
    Returns None
    '''
    if not mrt_stations:
        mrt_stations = ['Admiralty MRT', 'Aljunied MRT', 'Ang Mo Kio MRT', 'Bakau LRT', 'Bangkit LRT', 'Bartley MRT', 'Bayfront MRT',
                        'Bayshore MRT', 'Beauty World MRT', 'Bedok MRT', 'Bedok North MRT', 'Bedok Reservoir MRT', 'Bencoolen MRT',
                        'Bendemeer MRT', 'Bishan MRT', 'Boon Keng MRT', 'Boon Lay MRT', 'Botanic Gardens MRT', 'Braddell MRT',
                        'Bras Basah MRT', 'Buangkok MRT', 'Bugis MRT', 'Bukit Batok MRT', 'Bukit Brown MRT', 'Bukit Gombak MRT',
                        'Bukit Panjang MRT', 'Buona Vista MRT', 'Caldecott MRT', 'Cashew MRT', 'Changi Airport MRT',
                        'Chinatown MRT', 'Chinese Garden MRT', 'Choa Chu Kang MRT', 'City Hall MRT', 'Clarke Quay MRT',
                        'Clementi MRT', 'Commonwealth MRT', 'Compassvale LRT', 'Cove LRT', 'Dakota MRT', 'Dhoby Ghaut MRT',
                        'Downtown MRT', 'Xilin MRT', 'Tampines East MRT', 'Mayflower MRT', 'Upper Thomson MRT',
                        'Lentor MRT', 'Woodlands North MRT', 'Woodlands South MRT', 'Esplanade MRT', 'Eunos MRT',
                        'Expo MRT', 'Fajar LRT', 'Farmway LRT', 'Farrer Park MRT', 'Fort Canning MRT',
                        'Gardens by the Bay MRT', 'Geylang Bahru MRT', 'HarbourFront MRT', 'Haw Par Villa MRT', 'Hillview MRT',
                        'Holland Village MRT', 'Hougang MRT', 'Jalan Besar MRT', 'Joo Koon MRT', 'Jurong East MRT',
                        'Jurong West MRT', 'Kadaloor LRT', 'Kaki Bukit MRT', 'Kallang MRT', 'Kembangan MRT', 'Keppel MRT',
                        'King Albert Park MRT', 'Kovan MRT', 'Kranji MRT', 'Labrador Park MRT', 'Lakeside MRT', 'Lavender MRT',
                        'Layar LRT', 'Little India MRT', 'Lorong Chuan MRT', 'MacPherson MRT', 'Marina Bay MRT', 'Marina South Pier MRT',
                        'Marsiling MRT', 'Marymount MRT', 'Mattar MRT', 'Meridian LRT', 'Mountbatten MRT',
                        'Newton MRT', 'Nibong LRT', 'Nicoll Highway MRT', 'Novena MRT', 'Oasis LRT', 'One-North MRT', 'Orchard MRT',
                        'Outram Park MRT', 'Paya Lebar MRT', 'Pasir Ris MRT', 'Paya Lebar MRT', 'Pasir Ris MRT', 'Paya Lebar MRT', 'Pasir Ris MRT', 
                        'Pioneer MRT', 'Potong Pasir MRT', 'Promenade MRT', 'Punggol MRT', 'Queenstown MRT', 'Raffles Place MRT', 'Redhill MRT',
                        'Riviera LRT', 'Rochor MRT', 'Sembawang MRT', 'Sengkang MRT', 'Serangoon MRT', 'Simei MRT', 'Sixth Avenue MRT', 
                        'Somerset MRT', 'Springleaf MRT', 'Stadium MRT', 'Stevens MRT', 'Sumang LRT', 'Tai Seng MRT', 'Tampines MRT', 
                        'Tampines East MRT', 'Tampines West MRT', 'Tanah Merah MRT', 'Tanjong Pagar MRT', 'Tanjong Rhu MRT', 'Teck Lee LRT', 
                        'Telok Ayer MRT', 'Telok Blangah MRT', 'Thanggam LRT', 'Tiong Bahru MRT', 'Toa Payoh MRT', 
                        'Tuas Crescent MRT', 'Tuas Link MRT', 'Tuas West Road MRT', 'Ubi MRT', 'Upper Changi MRT', 
                        'Woodlands MRT', 'Woodlands South MRT', 'Woodlands North MRT', 'Yew Tee MRT', 'Yio Chu Kang MRT', 'Yishun MRT']
    # Future stations - 'Tampines North MRT', 'Tengah MRT'

    mrt_coordinates = {}
    for mrt in mrt_stations:
        response = requests.get(f"https://www.onemap.gov.sg/api/common/elastic/search?searchVal={mrt}&returnGeom=Y&getAddrDetails=Y")
        response.raise_for_status()
        data = response.json()
        # string (lat,long) as key
        # mrt_coordinates[f"{data['results'][0]['LATITUDE']},{data['results'][0]['LONGITUDE']}"] = mrt
        mrt_coordinates[mrt] = (float(data['results'][0]['LATITUDE']),float(data['results'][0]['LONGITUDE']))
        
    with open(filepath, 'w')as f:
        json.dump(mrt_coordinates, f, indent=4)

@timeit
@error_handler
def load_mrt_coordinates(filepath: str) -> dict:
    '''
    Function to read saved mrt_coordinates from json file
    ## Parameters
    filepath : str
        filepath to json file
    Returns data : dictionary
    '''
    with open(filepath, 'r') as f:
        file = f.read()
        data = json.loads(file)
        return data


@error_handler
def find_nearest_stations(geo_data_df : pd.DataFrame, mrt_stations : np.array, mrt_coordinates : np.array, 
                          n_nearest_stations: int=2, verbose : int=0) -> list:
    '''
    Function to determine nearest MRT station of the resale_flat based on latitude and longitude
    ## Parameters
        geo_data_df : pd.DataFrame
        mrt_stations : np.array
        mrt_coordinates : np.array
        n_nearest_stations: int=2
        verbose : int=0

    Returns a list of n_nearest stations
    '''
    # Matrix substraction to get difference with each MRT, convert to absolute values
    distance_diff = geo_data_df['numpy_array'] - mrt_coordinates
    absolute_dist = np.absolute(distance_diff)

    # Matrix sum over latitude and longitude of each entry
    sum_of_distances = np.sum(absolute_dist, axis=1)

    # Sort and search based on desired n_nearest_stations
    sorted_distances = np.sort(sum_of_distances)
    nearest_stations = []
    for n in range(n_nearest_stations):
        idx = np.where(sum_of_distances==sorted_distances[n])
        from_coordinates = tuple(geo_data_df['numpy_array'])
        to_coordinates = tuple(mrt_coordinates[idx][0])
        geodesic_dist = GD(from_coordinates, to_coordinates).kilometers
        nearest_stations.append(mrt_stations[idx][0])
        nearest_stations.append(np.round(geodesic_dist,2))

    if verbose==1:
        logger.info(f'Difference in distances: \n{distance_diff[:5]}')
        logger.info()
        logger.info(f'Absolute difference: \n{absolute_dist[:5]}')
        logger.info()
        logger.info(f'Sum of distances \n {sum_of_distances[:5]}')
        logger.info()
        logger.info(f'Sorted distances\n{sorted_distances[:5]}')
        logger.info()
        logger.info(f'Top {n_nearest_stations}')
        logger.info(nearest_stations)

    return nearest_stations

if __name__ ==  '__main__':
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
        
        # if config['automation'] & datetime.now().day != 30:
        #     etl_logger.info('Exiting ETL script - script will only run on 30th of each month')
        #     sys.exit()

        # Accounts for filepathing local and in pythonanywhere
        if config['local']:
            cache_filepath = config['local_cache_filepath']
        else:
            os.chdir(config['web_directory'])
            cache_filepath = 'project_cache'
        
        # files to append to
        output_file_train = config['train']
        output_file_test = config['test']

        # Determines whether to extract all data for current year, or particular year and months
        use_curr_datetime = config['use_datetime']
        if use_curr_datetime:
            timestamp = datetime.now()
            years = [timestamp.year]
            months = [x for x in range(1, timestamp.month+1)]
        else:
            years = config['years']
            months = config['months']

    # Get the correct etl_logger
    etl_logger = add_custom_logger('etl', file_path='logs/etl.log')

    etl_logger.info(f"{'-'*50}New ETL run started {'-'*50}")
    etl_logger.info(f'Data extraction settings:')
    etl_logger.info(f'\tuse_curr_datetime: {use_curr_datetime}')
    etl_logger.info(f'\tyear(s): {years}')
    etl_logger.info(f'\tmonth(s): {months}')

    # Enable caching
    session = requests_cache.CachedSession(cache_filepath, backend="sqlite")

    # There is now a limit to the API calls, so split to individual call for each month instead
    df = pd.DataFrame()
    etl_logger.info('Making API calls to data.gov.sg')
    for year in years:
        for month in months:
            temp_df = datagovsg_api_call_v2(year=year, month=month)
            etl_logger.info(f'\tData df shape received: {temp_df.shape}')
            if df.empty:
                df = temp_df
            else:
                df = pd.concat([df, temp_df])
    etl_logger.info('\t\tCompleted')

    # Data transformation and geolocationing
    etl_logger.info('Cleaning data')
    df = clean_df(df)
    etl_logger.info('\t\tCompleted')

    etl_logger.info('Getting geolocations')
    geo_data_df= get_location_data(df[['address']], verbose=1, cached_session=session)
    etl_logger.info('\t\tCompleted')

    etl_logger.info('Getting distances to city center (Marina Bay)')
    dist_to_marina_bay = multiple_distance_to(geo_data_df['numpy_array'], 'Marina Bay', dist_type='geodesic', verbose=1)
    dist_to_marina_bay = pd.Series(dist_to_marina_bay, name='dist_to_marina_bay')
    etl_logger.info('\t\tCompleted')

    etl_logger.info('Combining geolocation data to main')
    df = pd.concat([df, dist_to_marina_bay, geo_data_df['latitude'], geo_data_df['longitude'], geo_data_df['postal_code']], axis=1)
    etl_logger.info('\t\tCompleted')
    
    # Convert coordinates into numpy arrays
    mrt_coordinates_dict = load_mrt_coordinates('static/mrt_dict.json')
    mrt_stations = np.array(list(mrt_coordinates_dict.keys()))
    mrt_coordinates = np.array(list(mrt_coordinates_dict.values()))

    n_nearest_stations = 1
    # Matrix operations to find nearest MRT stations for each row
    etl_logger.info(f'Finding nearest stations: n={n_nearest_stations}')
    nearest_stations = geo_data_df.apply(find_nearest_stations, mrt_stations= mrt_stations, mrt_coordinates=mrt_coordinates, n_nearest_stations=n_nearest_stations, axis=1, verbose=0)
    nearest_stations_df = pd.DataFrame(nearest_stations.tolist(), index=geo_data_df.index, columns=['nearest_station_'+ str(x) for x in range(n_nearest_stations)] + ['dist_to_station_'+ str(x) for x in range(n_nearest_stations)])
    df = pd.concat([df, nearest_stations_df], axis=1)
    etl_logger.info('\t\tCompleted')

    etl_logger.info('Splitting data')
    year_month = sorted(df['year_month'].unique())
    etl_logger.info('\t\tTime range found:')
    etl_logger.info(year_month)

    # Save data
    csv_file = f'static/from_{year_month[0]}_to_{year_month[-1]}.csv'
    df.to_csv(csv_file)
    etl_logger.info(f'\t\tFull data saved as "{csv_file}" @ {datetime.now()}')

    # Split out most recent month as Test data, the rest as training data
    test = df[df['year_month']==year_month[-1]] 
    train = df[df['year_month']!=year_month[-1]] 

    train.to_csv(output_file_train)
    etl_logger.info(f'\t\tTraining data saved as {output_file_train} @ {datetime.now()}')
    test.to_csv(output_file_test)
    etl_logger.info(f'\t\tTest data saved as {output_file_test} @ {datetime.now()}')
    