#!/usr/local/bin/python

"""
Test for errors in the use of the classifier functions
"""

import numpy as np

def catch_input_error(Xtrn,Ytrn):
    '''Checks that the inputs have the correct shape. Specifically that the
    Ys are nx1 and the Xs are nxm for n,m integer'''
    assert Xtrn.shape[0] == Ytrn.shape[0], '''Xtrn & Ytrn should have the
    same number of rows'''
#    assert Xvl_tst.shape[0] == Yvl_tst.shape[0], ''' Xvl_tst & Yvl_tst should have the
#    same number of rows'''
    assert Xtrn.ndim > 1, '''Xtrn should have a non-zero number of
    columns'''
#    assert Xvl_tst.ndim > 1, '''Xvl_tst should have a non-zero number
#    of columns'''
    assert Ytrn.ndim == 1, '''Ytrn should have one column'''
#    assert Yvl_tst.ndim == 1, '''Yvl_tst should have one column'''

if __name__ == '__main__':
    # Test that it catches the errors
    from numpy import array as ar
    a = [1,1]
    print("Correct call of the function")
    catch_input_error(ar([a,a]),ar([1,1]))



