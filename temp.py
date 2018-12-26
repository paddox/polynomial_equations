from __future__ import absolute_import
from __future__ import print_function
import pyopencl as cl
import pyopencl.array as cl_array
import numpy
import numpy.linalg as la
import time

class Profiler(object):
    def __enter__(self):
        self._startTime = time.time()
         
    def __exit__(self, type, value, traceback):
        print("Elapsed time: {:.3f} sec".format(time.time() - self._startTime))


with Profiler() as p:
    a = numpy.array([0, -1]).astype(numpy.int32)
    b = numpy.array([0,-5,0,1]).astype(numpy.int32)

    platform = cl.get_platforms()[0]
    device = platform.get_devices()[0]
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx)

    a_dev = cl_array.to_device(queue, a)
    b_dev = cl_array.to_device(queue, b)
    sizea_dev = cl_array.to_device(queue, numpy.array([a.size], dtype=numpy.int32))
    sizeb_dev = cl_array.to_device(queue, numpy.array([b.size], dtype=numpy.int32))
    dest_dev = cl_array.empty_like(cl_array.to_device(queue, numpy.zeros(a.size + b.size - 1, dtype=numpy.int32)))

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

    '''
    smfunc = cl.elementwise.ElementwiseKernel(ctx, "float *a, float *b, float *c", 
                                            "c[i] = a[i] + b[i]","smfunc")
    '''

    prg.mul(queue, a.shape, None, a_dev.data, b_dev.data, sizea_dev.data, sizeb_dev.data, dest_dev.data)

    #print(dest_dev, sub)
    print(numpy.trim_zeros(dest_dev.get(), 'b'))

with Profiler() as p:
    def possible_solutions(polynom):
        from fractions import Fraction
        numerator = get_deviders(abs(polynom[0]))
        denomerator = get_deviders(abs(polynom[-1]))
        solutions = numpy.zeros(numerator.size * denomerator.size, dtype=object)
        j = 0
        for t in numerator:
            for s in denomerator:
                solutions[j] = Fraction(t, s)
                j += 1
        return numpy.unique(numpy.trim_zeros(solutions, 'b'))
    def get_deviders(num):
        from math import sqrt
        ret = numpy.zeros(2 * num, dtype=numpy.int32)
        j = 0
        for i in range(1, int(sqrt(num)) + 1):
            if num % i == 0:
                ret[j] = i
                j += 1
                ret[j] = -i
                j += 1
                if num != i ** 2:
                    ret[j] = num // i
                    j += 1
                    ret[j] = -num // i
                    j += 1
        return numpy.trim_zeros(ret, 'b')
    def exam_solutions(polynom, solutions):
        ret = numpy.zeros(solutions.size, dtype=object)
        j = 0
        for solution in solutions:
            if sum([polynom[i] * solution ** i for i in range(len(polynom))]) == 0:
                ret[j] = solution
                j += 1
        return numpy.trim_zeros(ret, 'b')
    print(exam_solutions([-1,0,1],possible_solutions(numpy.array([-1,0,1]))))

