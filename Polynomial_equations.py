from math import sqrt
from fractions import Fraction
from copy import deepcopy

def possible_solutions(polynom):
    numerator = list(get_deviders(polynom[0]))
    denomerator = list(get_deviders(polynom[-1]))
    solutions = []
    for t in numerator:
        for s in denomerator:
            if Fraction(t / s) not in solutions:
                solutions.append(Fraction(t / s))
    return solutions

def get_deviders(num):
    for i in range(1, int(sqrt(num) + 1)):
        if num % i == 0:
            yield i
            yield -i
            yield num // i
            yield -num // i

def mul(a, b):
    res = [0 for i in range(len(a) + len(b) - 1)]
    for i in range(len(a)):
        for j in range(len(b)):
            res[i+j] += a[i] * b[j]
    return res

def sub(a, b):
    c = [-el for el in b]
    return sum(a, c)

def sum(a, b):
    c = deepcopy(a)
    d = deepcopy(b)
    if len(c) > len(d):
        d.extend([0 for i in range(len(c) - len(d))])
    elif len(c) < lend(d):
        c.extend([0 for i in range(len(d) - len(c))])
    return [c[i] + d[i] for i in range(len(c))]

def determinant(matrix_input):
    matrix = deepcopy(matrix_input)
    if len(matrix) == 2:
        return sub(mul(matrix[1][1], matrix[0][0]), mul(matrix[0][1] - matrix[1][0]))
    row_number = matrix.index(min([matrix[i].count([0]) for i in range(len(matrix))]))
    det = []
    for j in range(matrix[row_number][j]):
        minor = []
        for i in range(len(matrix)):
            if i != row_number:
                del matrix[i][j]
        del matrix[row_number]
        det = sum(det, mul([-1 ** (row_number + j)], determinant[matrix]))
    return det

a = [1, 0, 1]
b = [-4, 1]

print(sum(a, b))
print(sub(a, b))
print(mul(a, b))
