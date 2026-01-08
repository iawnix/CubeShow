import time
from functools import wraps
from rich import print as rp

def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        rp(f"Info\\[iaw]>: Function '{func.__name__}' executed in {end_time - start_time:.4f} seconds.")
        return result
    return wrapper