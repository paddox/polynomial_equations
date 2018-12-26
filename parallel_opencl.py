from __future__ import absolute_import
from __future__ import print_function
from math import sqrt
from fractions import Fraction
from copy import deepcopy
import time
import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array


def possible_solutions(polynom):
    numerator = get_deviders(abs(polynom[0]))
    denomerator = get_deviders(abs(polynom[-1]))
    solutions = []
    for t in numerator:
        for s in denomerator:
            if Fraction(t, s) not in solutions:
                solutions.append(Fraction(t, s))
    return solutions


def exam_solutions(polynom, solutions):
    ret = []
    for solution in solutions:
        if sum([polynom[i] * solution ** i for i in range(len(polynom))]) == 0:
            ret.append(solution)
    return ret


def get_deviders(num):
    a = np.array([num]).astype(np.int32)
    platform = cl.get_platforms()[0]
    device = platform.get_devices()[0]
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx)
    a_dev = cl_array.to_device(queue, a)
    dest_dev = cl_array.empty_like(cl_array.to_device(queue, np.zeros((2 * a[0]), dtype=np.int32)))
    prg = cl.Program(ctx, """
        __kernel void sum(__global const int *a, __global int *c)
        {
            int i = 1;
            int n = a[0];
            int j = 0;
            while(i <= sqrt((float) n))
            {
                if(n%i==0) {
                    c[j] = i;
                    j++;
                    c[j] = -i;
                    j++;
                    if (i != (n / i)) {
                        c[j] = n/i;
                        j++;
                        c[j] = -n/i;
                        j++;
                    }
                } 
                i++;
            }
        }
        """).build()
    prg.sum(queue, a.shape, None, a_dev.data, dest_dev.data)
    return np.trim_zeros(dest_dev.get(), 'b').tolist()

def mul(a_, b_):
    a = np.array(a_).astype(np.int32)
    b = np.array(b_).astype(np.int32)
    platform = cl.get_platforms()[0]
    device = platform.get_devices()[0]
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx)
    a_dev = cl_array.to_device(queue, a)
    b_dev = cl_array.to_device(queue, b)
    sizea_dev = cl_array.to_device(queue, np.array([a.size], dtype=np.int32))
    sizeb_dev = cl_array.to_device(queue, np.array([b.size], dtype=np.int32))
    dest_dev = cl_array.empty_like(cl_array.to_device(queue, np.zeros(a.size + b.size - 1, dtype=np.int32)))
    prg = cl.Program(ctx, """
        __kernel void mul(__global const int *a, __global const int *b, __global const int *sizea, __global const int *sizeb,  __global int *c)
        {
            int size_a = sizea[0];
            int size_b = sizeb[0];
            for(int i=0; i<size_a; i++) {
                for(int j=0; j<size_b; j++) {
                    c[i+j] += a[i] * b[j];
                }
            }
        }
        """).build()
    prg.mul(queue, a.shape, None, a_dev.data, b_dev.data, sizea_dev.data, sizeb_dev.data, dest_dev.data)

    #print(dest_dev, sub)
    return np.trim_zeros(dest_dev.get(), 'b').tolist()


def sub(a, b):

    a_np = np.array(a, dtype=np.int32)
    b_np = np.array(b, dtype=np.int32)
    if a_np.size > b_np.size:
        b_np.resize(a_np.size, refcheck=False)
    else:
        a_np.resize(b_np.size, refcheck=False)
    platform = cl.get_platforms()[0]
    device = platform.get_devices()[0]
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx)
    a_dev = cl_array.to_device(queue, a_np)
    b_dev = cl_array.to_device(queue, b_np)
    dest_dev = cl_array.empty_like(a_dev)
    subfunc = cl.elementwise.ElementwiseKernel(ctx, "int *a, int *b, int *c", "c[i] = a[i] - b[i]", "subfunc")
    subfunc(a_dev, b_dev, dest_dev)

    return dest_dev.get().tolist()


def sum_(a, b):
    a_np = np.array(a, dtype=np.int32)
    b_np = np.array(b, dtype=np.int32)
    if a_np.shape > b_np.shape:
        b_np.resize(a_np.shape, refcheck=False)
    else:
        a_np.resize(b_np.shape, refcheck=False)
    platform = cl.get_platforms()[0]
    device = platform.get_devices()[0]
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx)
    a_dev = cl_array.to_device(queue, a_np)
    b_dev = cl_array.to_device(queue, b_np)
    dest_dev = cl_array.empty_like(a_dev)
    sumfunc = cl.elementwise.ElementwiseKernel(ctx, "int *a, int *b, int *c", "c[i] = a[i] + b[i]", "sumfunc")
    sumfunc(a_dev, b_dev, dest_dev)

    return dest_dev.get().tolist()


def determinant(matrix_input):
    matrix = deepcopy(matrix_input)
    '''
    max_size = max([max([len(elem) for elem in row]) for row in matrix_])
    i = 0
    for row in matrix_input:        
        j = 0
        for elem in row:
            while len(matrix_[i][j]) != max_size:
                matrix_[i][j].append(0)
            j += 1
        i += 1
    matrix = np.array(matrix_)
    print(matrix_)
    '''
    if len(matrix) != len(matrix[0]):
        raise Exception("There is not square matrix.")
    if len(matrix) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        det22 = sub(mul(matrix[1][1], matrix[0][0]), mul(matrix[0][1], matrix[1][0]))
        #print("det of 2x2 matrix {0} is {1}".format(matrix, det22))
        return det22
    #row_number = matrix.index(min([matrix[i].count([0]) for i in range(len(matrix))]))
    row_number = 0
    det = []
    for j in range(len(matrix_input[row_number])):
        if matrix_input[row_number][j] == [0]:
            continue
        dfgdffg = [[matrix[l][k] for k in range(len(matrix[0])) if k!=j] for l in range(len(matrix)) if l!=row_number]
        det = sum_(
                   det, 
                   mul(
                       mul(
                           [(-1) ** (row_number + j)], 
                           matrix[row_number][j]
                          ),
                       determinant([[matrix[l][k] for k in range(len(matrix[0])) if k!=j] for l in range(len(matrix)) if l!=row_number])
                      )
                  )
    return det


def computeGCD(x, y):   
   while(y): 
       x, y = y, x % y   
   return x 


def computeLCM(x, y):
    return x * y // computeGCD(x, y)


def list_LCM(array):
    gcd = array[0]
    for i in range(len(array)-1):
        gcd = computeLCM(gcd, array[i+1])
    return gcd

def read_input():
    f = open('input.txt')
    equation = []
    mtrx = [[], []]
    num_eq = 0
    for line in f:
        equation.append((list(map(Fraction, line.split(', ')))))
        #print(equation[num_eq])
        lcm = list_LCM([koeff.denominator for koeff in equation[num_eq]])
        #print(lcm)
        equation[num_eq] = [int(koeff * lcm) for koeff in equation[num_eq]]
        #print(equation[num_eq])
        koeff_len = int(sqrt(len(equation[num_eq])))
        for i in range(koeff_len):
            mtrx[num_eq].append(equation[num_eq][i*koeff_len:(i+1)*koeff_len])
            mtrx[num_eq][i].reverse()
        num_eq += 1
    return equation, mtrx

def make_mtrx(equation):
    mtrx = [[], []]
    for num_eq in range(2):
        koeff_len = int(sqrt(len(equation[num_eq])))
        for i in range(koeff_len):
            mtrx[num_eq].append(equation[num_eq][i*koeff_len:(i+1)*koeff_len])
            mtrx[num_eq][i].reverse()
    return mtrx

def make_resultant_matrix(equation, mtrx):
    d1 = np.zeros
    d1 = int(sqrt(len(equation[0]))) - 1
    d2 = int(sqrt(len(equation[1]))) - 1
    #print(d1, d2)
    resultant_mtrx = []
    d1_index = d1
    d2_index = d2
    while d2_index - 1 >= 0:
        row = []
        j = 0
        for i in range(d1+d2):        
            if i in range(d2_index-1, d1_index+d2_index):
                row.append(mtrx[0][j])
                j += 1
            else:
                row.append([0])
        resultant_mtrx.append(row)
        d2_index -= 1

    d1_index = d1
    d2_index = d2
    while d1_index - 1 >= 0:
        row = []
        j = 0
        for i in range(d1+d2):        
            if i in range(d1_index-1, d1_index+d2_index):
                row.append(mtrx[1][j])
                j += 1
            else:
                row.append([0])
        resultant_mtrx.append(row)
        d1_index -= 1
    return resultant_mtrx


def exam_x(solutions_y, equation):
    for solut in solutions_y:
        #print("{0}/{1}".format(solut.numerator, solut.denominator))
        solut_x = []
        for i in range(len(equation)):
            max_power = int(sqrt(len(equation[i]))) - 1
            eq_x = [equation[i][j] * solut ** (max_power - (j % (max_power + 1))) for j in range(len(equation[i]))]
            #print(eq_x)
            polynom_x = [sum(eq_x[j: j+max_power+1]) for j in range(0, len(eq_x), max_power+1)]
            polynom_x.reverse()
            #print(polynom_x)
            poss_solut_x = possible_solutions(polynom_x)
            solut_x.append(exam_solutions(polynom_x, poss_solut_x))
        #print(solut_x)
        for i in range(len(solut_x)):
            solut_x[i] = list(solut_x[i])
        solut_x = list(map(set, solut_x))
        #print(solut_x)
        eq_solve_x = solut_x[0] & solut_x[1]
        for x in eq_solve_x:
            yield x, solut

def main():
    starttime = time.time()
    equation, mtrx = read_input()
    #print(equation)
    res_mtrx = make_resultant_matrix(equation, mtrx)
    det = determinant(res_mtrx)
    possible_solut = possible_solutions(det)
    soluts = exam_solutions(det, possible_solut)
    #exam_x(soluts)
    for solution in exam_x(soluts, equation):
        print("({0}, {1})".format(str(solution[0]), str(solution[1])))
    print(time.time() - starttime, "s")

def exam_main(equation):
    mtrx = make_mtrx(equation)
    print(equation, mtrx)
    res_mtrx = make_resultant_matrix(equation, mtrx)
    det = determinant(res_mtrx)
    possible_solut = possible_solutions(det)
    soluts = exam_solutions(det, possible_solut)
    #exam_x(soluts)
    for solution in exam_x(soluts, equation):
        print("({0}, {1})".format(str(solution[0]), str(solution[1])))

if __name__ == "__main__":
    eq = [[1, 0, 0, -1], [0, 0, 4, 0, 0, 0, 1, 0, -5]]


    for i in np.logspace(0, 10, num=11, dtype=int, base=2):
        starttime = time.time()
        exam_main([[el * i for el in equat] for equat in eq])
        print(time.time() - starttime, "s")

    







