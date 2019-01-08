from math import sqrt
from fractions import Fraction
from copy import deepcopy
import time
import numpy as np
from numba import jit, prange
import timeit
import cProfile

@jit(nopython=True, parallel=True)
def parallel_possible_solutions(polynom):
    numerator = get_deviders(abs(polynom[0]))
    denomerator = get_deviders(abs(polynom[-1]))
    solutions = np.zeros((numerator.size * denomerator.size, 2), dtype=np.int64)
    j = 0
    for t in numerator:
        for s in denomerator:
            gcd = computeGCD(t, s)
            solutions[j][0] = t // gcd
            solutions[j][1] = s // gcd
            j += 1
    i = solutions.shape[0]-1
    while solutions[i][1] == 0 and solutions[i][0] == 0:
        i -= 1    
    return solutions[:i]

def possible_solutions(polynom):
    ret = np.unique(parallel_possible_solutions(polynom), axis=0)
    ret1 = np.zeros(ret.shape[0], dtype=object)
    for i in range(ret1.shape[0]):
        ret1[i] = Fraction(ret[i][0], ret[i][1])
    return np.unique(ret1)

#@jit(nopython=True, parallel=True)
def np_exam_solutions(polynom, np_solutions):
    ret = []
    for solution in np_solutions:
        sum_ = 0.0
        for i in prange(polynom.shape[0]):
            sum_ += polynom[i] * ((solution[0] / solution[1]) ** i)
        if sum_ < 0.1 and sum_ > -0.1:
            ret.append(solution)
    return ret

def exam_solutions(polynom, solutions):
    np_solutions = np.zeros((solutions.shape[0], 2), dtype=np.int64)
    for i in range(solutions.shape[0]):
        np_solutions[i][0] = solutions[i].numerator 
        np_solutions[i][1] = solutions[i].denominator
    ret = np_exam_solutions(polynom, np_solutions)
    ret1 = np.zeros(len(ret), dtype=object)
    for i in range(ret1.shape[0]):
        ret1[i] = Fraction(ret[i][0], ret[i][1])
    return ret1

@jit(nopython=True, parallel=True)
def get_deviders(num):
    ret = np.zeros(2 * num, dtype=np.int64)
    j = 0
    i = 1
    while i <= num:
        if num % i == 0:
            ret[j] = i
            ret[j+1] = -i
            j += 2
        i += 1
    i = 2 * num - 1
    while ret[i] == 0:
        i -= 1
    return ret[:i+1]

@jit(nopython=True, parallel=True)
def mul(a, b):
    res = np.zeros(a.shape[0]+b.shape[0]-1, dtype=np.int64)
    for i in range(a.shape[0]):
        for j in range(b.shape[0]):
            res[i+j] += a[i] * b[j]
    i = a.shape[0]+b.shape[0]-2
    while res[i] == 0:
        i -= 1
    return res[:i+1]

#@jit(nopython=True, parallel=True)
def determinant(matrix):
    det = np.array([0])
    if matrix.shape[0] == 1:
        return matrix[0][0]
    if matrix.shape[0] == 2:
        mul1 = mul(matrix[0, 0], matrix[1, 1])
        mul2 = mul(matrix[0, 1], matrix[1, 0])
        diff = mul1.shape[0] - mul2.shape[0]
        zero = np.zeros(abs(diff), dtype=np.int64)
        if diff > 0:
            mul2 = np.append(mul2, zero)
        if diff < 0:
            mul1 = np.append(mul1, zero)
        det22 = mul1 - mul2
        #print(det22)
        return det22
    i = 0
    j = 0
    while i < matrix.shape[0]:
        flag = 1
        while flag:
            i += 1
            for k in range(matrix.shape[2]):
                if matrix[i-1,j,k] != 0:
                    i -= 1
                    flag = 0
                    break
        minor = np.delete(np.delete(matrix, i, 0), j, 1)
        mul1 = mul(determinant(minor), mul(np.array([(-1) ** (i + j)]), matrix[i,j]))
        diff = mul1.shape[0] - det.shape[0]
        zero = np.zeros(abs(diff), dtype=np.int64)
        if diff > 0:
            det = np.append(det, zero)
        if diff < 0:
            mul1 = np.append(mul1, zero)
        det = det + mul1
        i += 1
    return det

@jit(nopython=True, parallel=True)
def computeGCD(x, y):   
   while(y): 
       x, y = y, x % y   
   return x 

@jit(nopython=True, parallel=True)
def computeLCM(x, y):
    return x * y // computeGCD(x, y)

@jit(nopython=True, parallel=True)
def list_LCM(array):
    lcm = array[0]
    for i in range(array.shape[0]-1):
        lcm = computeLCM(lcm, array[i+1])
    return lcm

@jit(nopython=True, parallel=True)
def list_GCD(array):
    gcd = array[0]
    for i in range(array.shape[0]-1):
        gcd = computeGCD(gcd, array[i+1])
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

@jit(nopython=True, parallel=True)
def make_resultant_matrix(mtrx):
    d1 = mtrx[0].shape[0] - 1
    d2 = mtrx[1].shape[0] - 1
    koeff_size = max(d1, d2)+1
    #print(d1, d2)
    resultant_mtrx = np.zeros((d1+d2, d1+d2, koeff_size), dtype=np.int64)
    look_row = np.zeros((d1+2*d2-1, koeff_size), dtype=np.int64)
    for i in prange(d2-1, d1+d2-1+1):
        size = mtrx[0].shape[1]
        look_row[i][:size] = mtrx[0][i-(d2-1)]
    for i in prange(d2):
        resultant_mtrx[i] = look_row[i:i+d1+d2]
    look_row2 =np.zeros((2*d1+d2-1, koeff_size), dtype=np.int64)
    for i in prange(d1-1, d1+d2-1+1):
        size = mtrx[1].shape[1]
        look_row2[i][:size] = mtrx[1][i-(d1-1)]
    for i in prange(d2, d1+d2):
        resultant_mtrx[i] = look_row2[i-d2:i+d1]
    return resultant_mtrx

def exam_x(solutions_y, equation):
    equation[0] = np.transpose(np.flipud(equation[0]))
    equation[1] = np.transpose(np.flipud(equation[1]))
    for solution in solutions_y:
        solut_x = []
        eq = []
        for i in range(2):
            equation_xy = np.zeros(equation[i].shape, dtype=object)
            for j in range(equation[i].shape[0]):                
                equation_xy[j] = equation[i][j] * (solution ** j)
            eq_x = np.sum(equation_xy, axis=0)
            eq_x_int = np.zeros((eq_x.shape[0], 2), dtype=np.int64)
            for j in range(eq_x_int.shape[0]):
                eq_x_int[j][0] = eq_x[j].numerator
                eq_x_int[j][1] = eq_x[j].denominator
            eq.append(eq_x_int)
        eq1 = get_integer_equation(eq)
        poss_solut_x1 = possible_solutions(eq1[0])
        poss_solut_x2 = possible_solutions(eq1[1])
        solut_x.append(exam_solutions(eq1[0], poss_solut_x1))
        solut_x.append(exam_solutions(eq1[1], poss_solut_x2))
        for x in np.intersect1d(solut_x[0], solut_x[1]):
            yield x, solution

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


def convert_to_array(equation):
    converted_equation = [[], []]
    mtrx = [[], []]
    num_eq = 0
    for line in equation.split('\n'):
        for koeff in line.split(', '):
            koeff_rational = Fraction(koeff)
            numerator = koeff_rational.numerator
            denominator = koeff_rational.denominator
            converted_equation[num_eq].append([numerator, denominator])
        num_eq += 1
    converted_equation[0] = np.array(converted_equation[0])
    converted_equation[1] = np.array(converted_equation[1])
    return converted_equation

@jit(nopython=True, parallel=True)
def get_integer_equation(equation):
    integer_equation = []
    num = 0
    for i in prange(2):
        int_equation = np.zeros(equation[i].shape[0], dtype=np.int64)
        print()
        denominators = equation[i][:, 1]
        lcm = list_LCM(denominators)
        for j in prange(int_equation.shape[0]):
            int_equation[j] = lcm // equation[i][j][1] * equation[i][j][0]
        integer_equation.append(int_equation)
    return integer_equation

@jit(nopython=True, parallel=True)     
def sqrt_int(a):
    for i in prange(1000):
        if i ** 2 == a:
            return i

def reverse(equation):
    reverse_eq = []
    for eq in equation:
        reverse_eq.append(np.flip(eq, 1))
    return reverse_eq

def new_main(eq):
    #eq = "500, 0, 0, -500\n0, 0, 2000, 0, 0, 0, 500, 0, -2500"
    #print(eq)
    starttime = time.time()
    converted_eq = convert_to_array(eq)
    #print(converted_eq)
    int_eq = get_integer_equation(converted_eq)
    #print(int_eq)
    power = [sqrt_int(int_eq[0].shape[0]), sqrt_int(int_eq[1].shape[0])]
    int_eq[0].resize((power[0], power[0]))
    int_eq[1].resize((power[1], power[1]))
    reverse_eq = reverse(int_eq)
    #print("transpose",np.transpose(np.flipud(reverse_eq[0])), np.transpose(np.flipud(reverse_eq[1])))
    res_mtrx = make_resultant_matrix(reverse_eq)
    det = determinant(res_mtrx)
    gcd = list_GCD(det)
    if gcd != 1:
        for i in range(det.shape[0]):
            det[i] = det[i] // gcd
    #print("{0:.20f} s".format(time.time() - starttime))
    possible_solut = possible_solutions(det)
    soluts = exam_solutions(det, possible_solut)
    #exam_x(soluts)
    #print(soluts)
    for solution in exam_x(soluts, reverse_eq):
        print("({0}, {1})".format(str(solution[0]), str(solution[1])))
    print("{0:.20f} s".format(time.time() - starttime))


if __name__ == "__main__":    
    times = []
    loops = 20
    one = 1
    four = 4
    five = 5
    eq = "{0}, 0, 0, -{0}\n0, 0, {1}, 0, 0, 0, {0}, 0, -{2}".format(one, four, five)
    new_main(eq)
    for i in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]:
        eq = "{0}, 0, 0, -{0}\n0, 0, {1}, 0, 0, 0, {0}, 0, -{2}".format(one*i, four*i, five*i)
        sec = timeit.timeit("new_main(eq)", setup="from __main__ import new_main, eq", number=loops)
        times.append(sec / loops)
    print(times)

    

    







