#!/usr/bin/env python3

from typing import List
import random
import sys
from PIL import Image
import pandas as pd
import numpy as np

def matrix_from_bin(bin_list):
	mat_size = len(bin_list)

	g = np.zeros(shape=(mat_size, mat_size))

	for i in range(mat_size):
		g[i][bin_list[i]] = 1

	return g

# input param is the amount of iterations
# python3 PolarCodes.py 3
def main():
	size = int(sys.argv[1])

	# https://oeis.org/A030109
	# https://stackoverflow.com/questions/70248207/numpy-generate-matrix-recursively
	# https://codegolf.stackexchange.com/questions/83373/bit-reversal-permutations
	b_n = lambda n:[int(bin(i+2**n)[:1:-1],2)//2 for i in range(2**n)]
	b_n_matrix = matrix_from_bin(b_n(size+1)).astype(int)
	print("B")
	print(b_n_matrix)

	g = np.array([[1, 0], [1, 1]])

	for i in range(size):
		g = (np.array([[g,np.zeros_like(g)],[g, g]])
		.swapaxes(1,2).reshape(2*g.shape[0], 2*g.shape[1])
		)
	print("G")
	print(g)
	print("BG")
	print(np.matmul(b_n_matrix, g))


if __name__ == "__main__":
	main()