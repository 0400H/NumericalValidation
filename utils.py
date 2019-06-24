# -*- coding:utf-8 -

import ctypes as ct
import numpy as np
import unittest

def matrix_validation(mat_a, mat_b):
    if mat_a.shape != mat_b.shape:
        print('mat_u8 shape != mat_s8 shape')
        return None

    shape_m = mat_a.shape[0]
    shape_n = mat_a.shape[1]
    length = shape_m * shape_n
    count = np.sum(mat_a != mat_b)
    return count, length