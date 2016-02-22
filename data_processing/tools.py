#!/usr/local/bin/python

import time


### Decorators
def timing_decorator(func):
    def wrapper(*args, **kwargs):
        t1 = time.time()
        output = func(*args, **kwargs)
        t2 = time.time()
        print 'Program Running Time: %.3f seconds.' % (t2 - t1)
        return output
    return wrapper

if __name__ == '__main__':
    pass
