import time
import functools
from qa.tools.logger import setup_logger

logger = setup_logger()

# class Singleton(type):
#     _instances = {}

#     def __call__(cls, *args, **kwargs):
#         key = str(cls) + str(args) + str(kwargs)
#         if key not in cls._instances:
#             cls._instances[key] = super(Singleton,
#                                         cls).__call__(*args, **kwargs)
#         return cls._instances[key]


def trycatch(func):
    """
    decorator for putting try-except block for a function
    Description:
        - If the function needs a try-except block, decorate it with this decorator
        - This decorator will keep your function in a try-except block and will report the exception
    Use:
        >>> from decorators import trycatch
        >>> @trycatch
        ... def func():
        ...     print(0/0) # Division by 0 must raise exception
        ...
        >>> func()
        Exception occurred: [integer division or modulo by zero]
        >>>
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            logger.exception('Exception occurred: [{}]'.format(e))

    return wrapper


def timer(func):
    """
    decorator for printing the time required by a function to execute
    Description:
        - If the function needs a timer to check how long it takes for completion, use this decorator
        - This decorator will print the time required by the function to which it's decorating
    Use:
        >>> from decorators import timer
        >>> @timer
        ... def a():
        ...     import time
        ...     print('Hi')
        ...     time.sleep(1)
        ...
        >>> a()
        Hi
        Time taken by the function is [1.00103902817] sec
        >>>
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        from time import time
        start_time = time()
        func(*args, **kwargs)
        end_time = time()
        logger.info('Time taken by the function is [{time}] sec'.format(
            func=func, time=end_time - start_time))

    return wrapper


def Singleton(cls):
    """
    decorator for creating a singleton class
    Description:
        - Decorate your class with this decorator
        - If you happen to create another instance of the same class, it will return the previously created one
        - Supports creation of multiple instances of same class with different args/kwargs
        - Works for multiple classes
    Use:
        >>> from decorators import singleton
        >>>
        >>> @singleton
        ... class A:
        ...     def __init__(self, *args, **kwargs):
        ...         pass
        ...
        >>>
        >>> a = A(name='Siddhesh')
        >>> b = A(name='Siddhesh', lname='Sathe')
        >>> c = A(name='Siddhesh', lname='Sathe')
        >>> a is b  # has to be different
        False
        >>> b is c  # has to be same
        True
        >>>
    """
    previous_instances = {}

    @functools.wraps(cls)
    def wrapper(*args, **kwargs):
        if cls in previous_instances and previous_instances.get(
                cls, None).get('args') == (args, kwargs):
            return previous_instances[cls].get('instance')
        else:
            previous_instances[cls] = {
                'args': (args, kwargs),
                'instance': cls(*args, **kwargs)
            }
            return previous_instances[cls].get('instance')

    return wrapper


def flatten(nested_list):
    """嵌套list 摊平: [1, [2, 3]] -> [1, 2, 3]"""
    queue = [nested_list]
    ret = []
    while queue:
        for item in queue.pop():
            if isinstance(item, list):
                queue.append(item)
            else:
                ret.append(item)
    return ret


# 打印程序运行时长
class PrintTime:
    def __init__(self, func):
        self.__func = func

    def __call__(self, *args, **kwargs):
        t1 = time.process_time()
        result = self.__func(*args, **kwargs)
        t2 = time.process_time()
        logger.debug('函数运行{}秒。'.format(t2 - t1))
        return result