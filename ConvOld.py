#!/usr/bin/env python3
from functools import reduce
from typing import List
import random
import sys
from PIL import Image
import numpy as np

# Programmes a convolutional encoder without denominators (only polynomials)
# Takes as input four params
# Number of input sequences (k, number of rows of matrix)
# Number of output sequences (n, number of columns of matrix)
# Basic matrix b (Separator is S)
# Message m
# Output: Encoding
# Another output: New message making the memory zero
# If k>1, then the message has a separator as well (S)

def multiply(a, b):
    sum = 0

    ind_counter = 0
    while b != 0:
        if b % 2 != 0:
            adding = (a << ind_counter)
            sum = sum ^ adding
        ind_counter += 1
        b = b >> 1
    return sum

# Example:
# python ConvOld.py k n b m
# python ConvOld.py 1 2 1111S1011 1011011
# Which represents
# One input, two outputs, Matrix (1+x+x^2+x^3, 1+x^2+x^3), Message: 1011011
def main():
    k = int(sys.argv[1])
    n = int(sys.argv[2])
    b_temp = sys.argv[3]
    m_temp = sys.argv[4]
    int('11111111', 2)

    b_temp = b_temp.split("S")
    b_temp_bin = []
    for i in b_temp:
        num = int(i[::-1], 2)
        b_temp_bin.append(num)
    
    b = np.array(b_temp_bin).reshape(k, n)
    #print(b)

    m_temp = m_temp.split("S")
    m_temp_bin = []
    for i in m_temp:
        num = int(i[::-1], 2)
        m_temp_bin.append(num)
    #print(m_temp_bin)

    m = np.array(m_temp_bin)
    print(m)

    # Checking dimensions
    if len(m) != len(b):
        print("Wrong dimensions")

    # End of input
    # Start of calculations
    b = np.transpose(b, (1,0))

    output_counter = 0
    output_strings = []
    output_length = 0
    for idx_b_col, b_col in enumerate(b):
        output_counter += 1
        sum = 0
        for idx_m, m_val in enumerate(m):
            c = m_val
            d = b_col[idx_m]
            # print("c: "+str("{0:b}".format(c))+"; d: "+str("{0:b}".format(d)))
            part_sum = reduce(lambda c,x: c^x, (c << i for i in range(len(bin(d))-2) if (d&(1<<i))>0),0)
            # print("Part sum: "+str("{0:b}".format(part_sum)))
            sum = sum ^ part_sum
        output_str = str("{0:b}".format(sum))[::-1]
        output_strings.append(output_str)
        output_length = len(output_str)
        print("c("+str(output_counter)+") = "+output_str)
    
    final_str = ""
    print(output_strings)
    for output_ind in range(output_length):
        for output_el in output_strings:
            final_str += output_el[output_ind]

    print("Overall string (length="+str(len(final_str))+")")
    print(final_str)

if __name__ == "__main__":
	main()