# -*- coding:utf-8 -

from utils import *

def MatMul_UnSn(mat_un, mat_sn, mat_s2n):
    np.matmul(mat_un, mat_sn, mat_s2n)
    # print(mat_un.shape, mat_sn.shape, mat_s2n.shape, mat_s2n)
    return None

def MatMul_SnSn(mat_a, mat_b, mat_s2n):
    if mat_a.dtype != mat_b.dtype:
        print('mat_a dtype != mat_b dtype')
        return None
    if len(mat_a.shape) != 2 or \
       len(mat_b.shape) != 2:
        print('pls input 2d matix')
        return None
    if mat_a.shape[1] != mat_b.shape[0]:
        print('mat_a dim1 != mat_b dim0')
        return None

    shape_m = mat_a.shape[0]
    shape_n = mat_b.shape[1]
    dtype = str(mat_a.dtype)
    bits = int(dtype[dtype.index('t')+1:])

    if shape_m >= shape_n:
        mat_a = mat_a + 2**(bits-1)
        mat_a = mat_a.astype(eval('np.uint'+str(bits)))
        mat_s2n_offset = np.zeros((1, shape_n), dtype=eval('np.int'+str(2*bits)))
        for index in range(shape_n):
            mat_s2n_offset[0, index] = sum(mat_b[:, index])
        mat_s2n_offset = -2**(bits-1) * mat_s2n_offset
        mat_s2n_offset = np.tile(mat_s2n_offset, (shape_m, 1))
    else:
        mat_b = mat_b + 2**(bits-1)
        mat_b = mat_b.astype(eval('np.uint'+str(bits)))
        mat_s2n_offset = np.zeros((shape_m, 1), dtype=eval('np.int'+str(2*bits)))
        for index in range(shape_m):
            mat_s2n_offset[index, 0] = sum(mat_a[index, :])
        mat_s2n_offset = -2**(bits-1) * mat_s2n_offset
        mat_s2n_offset = np.tile(mat_s2n_offset, (1, shape_n))

    MatMul_UnSn(mat_a, mat_b, mat_s2n)
    mat_s2n += mat_s2n_offset
    return None

def matmul_test(test_case, bits):
    (dim_m, dim_n, dim_k, value_a, value_b) = test_case
    mat_un = np.full((dim_m, dim_k), value_a, dtype=eval('np.uint'+str(bits)))
    mat_sn_1 = np.full((dim_m, dim_k), value_a, dtype=eval('np.int'+str(bits)))
    mat_sn_2 = np.full((dim_k, dim_n), value_b, dtype=eval('np.int'+str(bits)))
    # mat_s2n_ref = np.zeros((dim_m, dim_n), dtype=eval('np.int'+str(2*bits)))
    mat_s2n = np.zeros((dim_m, dim_n), dtype=eval('np.int'+str(2*bits)))
    mat_s2n_ref = np.matmul(mat_un, mat_sn_2)
    MatMul_SnSn(mat_sn_1, mat_sn_2, mat_s2n)
    count, length = matrix_validation(mat_s2n_ref, mat_s2n)
    # print('mat_s2n_ref', mat_s2n_ref.dtype, mat_s2n_ref)
    # print('mat_s2n', mat_s2n.dtype, mat_s2n)
    # print('error rate: %f' % (1.0 * count / length))
    return 1.0 * count / length

class unit_test(unittest.TestCase):
    target_error_rate = 0.0
    test_case = [
        (128, 64, 32, 0, 0),
        (128, 64, 32, 0, 64),
        (128, 64, 32, 64, 0),
        (128, 64, 32, 64, 64),
        (128, 64, 32, 64, -64),
        (128, 64, 32, 127, -128),
        (64, 128, 32, 127, -128),
        (64, 128, 128, 127, -128),
        (64, 128, 256, 127, -128),
        (64, 128, 512, 127, -128),
        (512, 512, 512, 127, -128),
    ]

    def test_8bit(self):
        for arg in self.test_case:
            error_rate = matmul_test(arg, 8)
            print('error_rate: %f' % error_rate)
            self.assertEqual(True, self.target_error_rate >= error_rate)

    def test_16bit(self):
        for arg in self.test_case:
            error_rate = matmul_test(arg, 16)
            print('error_rate: %f' % error_rate)
            self.assertEqual(True, self.target_error_rate >= error_rate)

    def test_32bit(self):
        for arg in self.test_case:
            error_rate = matmul_test(arg, 32)
            print('error_rate: %f' % error_rate)
            self.assertEqual(True, self.target_error_rate >= error_rate)

if __name__ == '__main__':
    unittest.main()