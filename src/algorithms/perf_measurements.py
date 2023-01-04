from functools import wraps
from time import perf_counter
from typing import Any, Callable


def timeit(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator for timing function execution time.

    Args:
        func: The function to time.

    Returns:
        The wrapped function.
    """

    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = perf_counter()
        result = func(*args, **kwargs)
        end_time = perf_counter()
        total_time = end_time - start_time
        print(f"Function {func.__name__} took {total_time:.2f} seconds")
        return result

    return timeit_wrapper


def return_runtime(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator that adds the execution time to the return values of the function.

    Args:
        func: The function to time.

    Returns:
        The wrapped function.
    """

    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = perf_counter()
        result = func(*args, **kwargs)
        end_time = perf_counter()
        total_time = end_time - start_time
        return total_time, result

    return timeit_wrapper


def checkpoint(time_ref: float = perf_counter()) -> Callable[..., None]:
    """
    Closure that stores a time checkpoint that is updated at every call.
    Each call prints the time elapsed since the last checkpoint with a custom message.

    Args:
        time_ref: The time reference to start from. By default, the time of the call will be taken.

    Returns:
        The closure.
    """

    def _closure(message: str = "") -> None:
        """
        Prints the time elapsed since the previous call.

        Args:
            message: Custom message to print. The overall result will be: 'message: time_elapsed'.
        """
        nonlocal time_ref
        current_time = perf_counter()
        if message != "":
            print(f"{message}: {current_time - time_ref:.4f}")
        time_ref = current_time

    return _closure
