import logging
import time
import sys
from requests.exceptions import HTTPError
from functools import wraps

def configure_logging(file_path=None, streaming=None, level=logging.INFO):
    '''
    Initiates the logger
    '''

    logger = logging.getLogger()
    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    if not len(logger.handlers):
        # Add a filehandler to output to a file
        if file_path:
            file_handler = logging.FileHandler(file_path, mode='w')
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        # Add a streamhandler to output to console
        if streaming:
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)    

    return logger

logger = configure_logging('logs/app.log', streaming=True)

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
        logger.info(f'{func.__name__}() started at {current_time} \t ended at \texecution time: {time_taken:.4f} seconds')
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
                logger.error(f'{func.__name__}() encountered {err}')
                # Raise exception if we reach max tries
                if i == max_attempts:
                    raise HTTPError(f'Exceeded max tries of {max_attempts}')

                # err.response gives us the Response object from requests module, we can call .status_code to get the code as int
                if err.response.status_code == 429:
                    logger.info(f'Sleeping for {delay} seconds', end = '\t')
                    time.sleep(delay)
                    logger.info('Retrying...', end='\t')
            except Exception as err:
                logger.error(f'{func.__name__}() encountered {err}') 
                break
            else:
                return result
    return error_handler_wrapper