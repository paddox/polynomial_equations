from math import sqrt
from fractions import Fraction
from copy import deepcopy
import time
from numba import jit, test
import numpy as np


def possible_solutions(polynom):
    if polynom[0]>0:
        numerator = get_deviders(polynom[0])
    else:
        numerator = get_deviders(-polynom[0])
    denomerator = get_deviders(abs(polynom[-1]))
    solutions = []
    for t in numerator:
        for s in denomerator:
            if Fraction(t / s) not in solutions:
                solutions.append(Fraction(t / s))
    return solutions

def exam_solutions(polynom, solutions):
    ret = []
    for solution in solutions:
        if sum([polynom[i] * solution ** i for i in range(len(polynom))]) == 0:
            ret.append(solution)
    return ret


def get_deviders(num):
    ret = []
    for i in range(1, int(sqrt(num)) + 1):
        if num % i == 0:
            ret.append(i)
            ret.append(-i)
            if num != i ** 2:
                ret.append(num // i)
                ret.append(-num // i)
    return ret

@jit(nopython=True, parallel=True)
def mul(a, b):
    res = [0]
    for i in range(len(a) + len(b) - 2):
        res.append(0)
    for i in range(len(a)):
        for j in range(len(b)):
            res[i+j] += a[i] * b[j]
    #print("mul {0} and {1}, result {2}".format(a, b, res))
    return res[:]

def sub(a, b):
    c = [-el for el in b]
    sb = sum_(a, c)
    #print("sub {0} and {1}, result {2}".format(a, b, sb))
    return sb

def sum_(a, b):
    #c = deepcopy(a)
    #d = deepcopy(b)
    res = []
    for i in range(max(len(a), len(b))):
        res.append(0)
    if len(a) > len(b):
        for i in range(len(a)):
            res[i] = a[i]
        for i in range(len(b)):
            res[i] += b[i]
    else:
        for i in range(len(b)):
            res[i] = b[i]
        for i in range(len(a)):
            res[i] += a[i]
    return res
    '''
    if len(c) > len(d):
        d.extend([0 for i in range(len(c) - len(d))])
    elif len(c) < len(d):
        c.extend([0 for i in range(len(d) - len(c))])
    #print("sum_ {0} and {1}, result {2}".format(a, b, [c[i] + d[i] for i in range(len(c))]))
    return [c[i] + d[i] for i in range(len(c))]
    '''


def determinant(matrix_input):
    matrix = [matrix_input[i] for i in range(len(matrix_input))]
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
        solut_x = list(map(set, solut_x))
        #print(solut_x)
        eq_solve_x = solut_x[0] & solut_x[1]
        for x in eq_solve_x:
            yield x, solut


def main():
    equation, mtrx = read_input()
    #print(equation)
    res_mtrx = make_resultant_matrix(equation, mtrx)
    det = determinant(res_mtrx)
    possible_solut = possible_solutions(det)
    soluts = exam_solutions(det, possible_solut)
    #exam_x(soluts)
    for solution in exam_x(soluts, equation):
        print("({0}, {1})".format(str(solution[0]), str(solution[1])))

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

    







