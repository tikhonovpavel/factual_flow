import functools
import torch




def argument_logger(arg=None):
    """Decorator that can be used with or without parenthesis and can be overridden at function call."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Determine whether to print based on decorator argument or function call override
            nonlocal arg
            arg = True if arg is None else arg  # Set default to True if arg is not provided
            should_print = kwargs.pop('argument_logger_do_print', arg)
            if should_print is False:
                return func(*args, **kwargs)

            # The rest of the decorator logic goes here
            arg_info = []
            arg_names = func.__code__.co_varnames[:func.__code__.co_argcount]

            # Helper function for argument representation
            def arg_to_string(name, value):
                if isinstance(value, (int, float, str)):
                    return f'{name}: {value}'
                elif isinstance(value, torch.Tensor):
                    if value.ndim <= 2 and value.size(0) <= 10 and (value.ndim == 1 or value.size(1) <= 10):
                        return f'{name}: {value.tolist()}'
                    else:
                        return f'{name}: Tensor shape: {value.shape}'
                else:
                    return f'{name}: {repr(value)}'

            # Combine and process arguments
            all_args = list(zip(arg_names, args)) + list(kwargs.items())
            for name, value in all_args:
                arg_info.append(arg_to_string(name, value))

            # Determine frame width and print
            max_arg_width = max(len(arg) for arg in arg_info)
            func_name_with_parentheses = func.__name__ + '()'
            max_width = max(max_arg_width, len(func_name_with_parentheses)) + 4  # Adjust for padding and function name length
            print('+' + '-' * (max_width - 2) + '+')
            print('| ' + func_name_with_parentheses.center(max_width - 4) + ' |')
            print('+' + '-' * (max_width - 2) + '+')
            for arg in arg_info:
                print('| ' + arg.ljust(max_width - 4) + ' |')
            print('+' + '-' * (max_width - 2) + '+')

            return func(*args, **kwargs)

        return wrapper

    # Check if the decorator is used without argument, i.e., @argument_logger
    if callable(arg):
        return decorator(arg)

    # If an argument is provided, return a decorator function
    return decorator


if __name__ == "__main__":
    @argument_logger
    def function_with_custom_class(custom_obj):
        pass

    @argument_logger
    def function_with_long_string(long_str):
        pass

    @argument_logger
    def function_with_multiple_args(a, b, c):
        pass

    @argument_logger(False)  # Configured not to print arguments
    def function_with_no_print(c, d):
        pass

    
    class CustomClass:
        def __init__(self, value):
            self.value = value


    custom_instance = CustomClass(value=123)
    function_with_custom_class(custom_instance)
    function_with_long_string("This is a very long string to test the argument logger decorator's ability to handle long strings effectively.")
    function_with_no_print(1, 2)
    function_with_no_print(3, 4, argument_logger_do_print=True)
    function_with_multiple_args(1, [1, 2, 3], {'key': 'value'})