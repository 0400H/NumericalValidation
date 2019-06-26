# -*- coding:utf-8 -

from utils import *

@jit
def MatMul(mat_a, mat_b, mat_out):
    shape_m = mat_a.shape[0]
    shape_k = mat_a.shape[1]
    shape_n = mat_b.shape[1]
    mat_b_trans = np.transpose(mat_b)
    for m_index in range(shape_m):
        for n_index in range(shape_n):
            for k_index in range(shape_k):
                mat_out[m_index, n_index] += mat_a[m_index, k_index] \
                                           * mat_b_trans[n_index, k_index]
    # print('MatMul Verbose:', mat_a[0], mat_b_trans, mat_out)
    return None

def MatMul_SnSn(mat_a, mat_b, mat_out):
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

    input_dtype = str(mat_b.dtype)
    output_dtype = str(mat_out.dtype)
    input_bits = int(input_dtype[input_dtype.index('t')+1: ])
    output_bits = int(output_dtype[output_dtype.index('t')+1: ])
    max_shape_k = 2**(output_bits - 2 * input_bits)
    shape_m = mat_a.shape[0]
    shape_n = mat_b.shape[1]
    shape_k = mat_a.shape[1]
    if shape_k > max_shape_k:
        print('data maybe overflow!')

    if shape_m >= shape_n:
        mat_a = mat_a + 2**(input_bits-1)
        mat_a = mat_a.astype(eval('np.uint'+str(input_bits)))
        mat_out_offset = np.zeros((1, shape_n), dtype=eval('np.int'+str(output_bits)))
        for n_index in range(shape_n):
            mat_out_offset[0, n_index] = np.sum(mat_b[:, n_index])
        # print('mat_out_offset', mat_out_offset.dtype, mat_out_offset)
        mat_out_offset = -2**(input_bits-1) * mat_out_offset
        mat_out_offset = np.tile(mat_out_offset, (shape_m, 1))
    else:
        mat_b = mat_b + 2**(input_bits-1)
        mat_b = mat_b.astype(eval('np.uint'+str(input_bits)))
        mat_out_offset = np.zeros((shape_m, 1), dtype=eval('np.int'+str(output_bits)))
        for index in range(shape_m):
            mat_out_offset[index, 0] = sum(mat_a[index, :])
        # print('mat_out_offset', mat_out_offset.dtype, mat_out_offset)
        mat_out_offset = -2**(input_bits-1) * mat_out_offset
        mat_out_offset = np.tile(mat_out_offset, (1, shape_n))

    # print('mat_out_offset', mat_out_offset.dtype, mat_out_offset)
    MatMul(mat_a, mat_b, mat_out)
    mat_out += mat_out_offset
    return None

def matmul_test(test_case, bits):
    (dim_m, dim_n, dim_k, value_a, value_b) = test_case
    mat_un = np.full((dim_m, dim_k), value_a, dtype=eval('np.uint'+str(bits)))
    mat_sn_1 = np.full((dim_m, dim_k), value_a, dtype=eval('np.int'+str(bits)))
    mat_sn_2 = np.full((dim_k, dim_n), value_b, dtype=eval('np.int'+str(bits)))
    # mat_sn_2 = np.random.randint(-2**(bits-1), 2**(bits-1), size=(dim_k, dim_n), dtype=eval('np.int'+str(bits)))
    # mat_sn_2 = np.random.randint(0, 2, size=(dim_k, dim_n), dtype=eval('np.int'+str(bits)))
    mat_out_ref = np.zeros((dim_m, dim_n), dtype=np.int64)
    mat_out = np.zeros((dim_m, dim_n), dtype=np.int64)
    MatMul(mat_un, mat_sn_2, mat_out_ref)
    MatMul_SnSn(mat_sn_1, mat_sn_2, mat_out)
    count, length = matrix_validation(mat_out_ref, mat_out)
    # print('mat_out_ref', mat_out_ref.dtype, mat_out_ref[0])
    # print('mat_out', mat_out.dtype, mat_out[0])
    return 1.0 * count / length

class unit_test(unittest.TestCase):
    target_error_rate = 0.0
    test_case = [
        (1, 1, 32, 64, 64),
        (128, 64, 1, 0, 0),
        (128, 64, 32, 0, 0),
        (128, 64, 32, 0, 64),
        (128, 64, 32, 64, 0),
        (128, 64, 32, 64, 64),
        (128, 64, 32, 64, -64),
        (128, 64, 32, 127, -128),
        (128, 64, 512, 127, -128),
        (64, 128, 32, 127, -128),
        (64, 128, 512, 127, -128),
    ]

    def test_8bit(self):
        bits = 8
        for arg in self.test_case:
            error_rate = matmul_test(arg, bits)
            result = self.target_error_rate >= error_rate
            print('bits: {}, testcse: {}, error_rate: {}'.format(bits, arg, error_rate))
            self.assertEqual(True, result)

    def test_16bit(self):
        bits = 16
        for arg in self.test_case:
            error_rate = matmul_test(arg, bits)
            result = self.target_error_rate >= error_rate
            print('bits: {}, testcse: {}, error_rate: {}'.format(bits, arg, error_rate))
            self.assertEqual(True, result)

    def test_32bit(self):
        bits = 32
        for arg in self.test_case:
            error_rate = matmul_test(arg, bits)
            result = self.target_error_rate >= error_rate
            print('bits: {}, testcse: {}, error_rate: {}'.format(bits, arg, error_rate))
            self.assertEqual(True, result)

if __name__ == '__main__':
    unittest.main()