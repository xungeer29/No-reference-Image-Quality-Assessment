import os
import numpy
sum_error = 0.0
num = 0
with open('test_result.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip('\n')
        error = line.split(':')[-1]
        error_abs = abs(float(error))
        sum_error += error_abs
        num += 1

    MAE = sum_error/num

    print MAE

