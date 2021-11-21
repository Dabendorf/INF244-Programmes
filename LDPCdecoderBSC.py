#!/usr/bin/env python3
from functools import reduce
from typing import Dict, List, Tuple
import sys
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy.stats import norm

def gallager(G: np.ndarray, H: np.ndarray, r: List[int], prob:float, debug_mode = False) -> List[int]:
	""" Gallagers sum product algorithm for AWGN
		G: matrix G of size nxk
		H: parity check matrix of size nxh
		r: received message
		N0: channel noise
		(Ec = 1)
	"""
	# Length of message
	n = len(r)

	# Calculate neighbourhood
	N_dict, M_dict = calc_neighbourhood(H=H)

	# List of non_empty entries in H
	non_empty_H_j_i = calc_non_zero(H=H)

	# Initialisation
	L_j = dict()
	L_j_i = dict()
	L_i_j = dict()
	vhat = dict()
	L_j_tot = dict()
	for j in range(n):
		# BSC
		a = ((-1) ** r[j])
		b = math.log(prob/(1-prob))
		L_j[j] = a * b

		for i in M_dict[j]:
			L_j_i[(j, i)] = L_j[j]
	
	codeword = False

	counter = 0
	while not codeword:
		counter += 1
		# Check node update
		# For every entry in H which is not zero
		# for (i,j)
		for (j, i) in non_empty_H_j_i:
			# Compute L_i_j
			L_i_j[(i, j)] = compute_checknode_gallager(N = N_dict[i], L_j_i = L_j_i, i=i, j=j, debug_mode=debug_mode)
		
		# Variable node update
		# print(non_empty_H_j_i)
		for (j, i) in non_empty_H_j_i:
			# Compute L_j_tot, vhat_j and L_j_i
			L_j_tot[j] = compute_l_j_tot(M = M_dict[j], L_i_j = L_i_j, j = j, Lj = L_j[j], debug_mode=debug_mode)
			vhat[j] = (np.sign(L_j_tot[j])+1) / 2
			L_j_i[(j, i)] = compute_l_j_i(M = M_dict[j], L_i_j = L_i_j, i = i, j = j, Lj = L_j[j])
		
		# Check termination condition
		vhat_list = np.array([vhat[key] for key in sorted(vhat.keys())])
		# vhat = (vhat_1 ... vhat_n)
		mult = H.dot(vhat_list)%2
		if np.sum(mult) == 0 or counter > 20:
			codeword = True

	return vhat_list
	"""G_ = np.identity(len(G))
	G2_ = np.zeros((len(G[0]) - len(G), len(G)))
	G_ = np.concatenate((G_, G2_))

	mhat = vhat_list.dot(G_)
	return mhat"""

def compute_checknode_gallager(N: List[int], L_j_i, i: int, j: int, debug_mode=False):
	"""Calculate the checknodes for the gallager algorithm"""
	temp = (-1) ** len(N) * 2

	atan_mult = 1
	for j2 in N:
		if j2 != j:
			atan_mult *= math.tanh(L_j_i[(j2, i)] / 2)

	return temp * math.atanh(atan_mult)

def compute_l_j_tot(M: List[int], L_i_j, j: int, Lj: float, debug_mode=False):
	"""Calculates l_j^tot for a given L_j"""
	sum = 0
	debug_str = "L_"+str(j)+"^tot = "+str(Lj)
	for i in M:
		debug_str += " + "+str(L_i_j[(i, j)])
		sum += L_i_j[(i, j)]

	if debug_mode:
		debug_str += " = "+str(sum + Lj)
		print(debug_str)

	return sum + Lj

def compute_l_j_i(M: List[int], L_i_j, i: int, j: int, Lj: float):
	sum = 0
	for i2 in M:
		if i2 != i:
			sum += L_i_j[(i2, j)]

	return sum + Lj


def calc_neighbourhood(H: np.ndarray) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
	""" Calculates the neighbourhood for each variable node and checknode and returns two dictionaries
		N(i): Set of variable nodes with edge to check node
		M(j): Set of check nodes with edge to variable node """
	N_dict = dict() # column numbers in every row being 1
	M_dict = dict() # row numbers in every column being 1

	# Calculate neighbourhood for check nodes
	for row_ind, row in enumerate(H):
		neighbour_list = []
		for col_ind, col in enumerate(row):
			if col == 1:
				neighbour_list.append(col_ind)
		N_dict[row_ind] = neighbour_list

	# Calculate neighbourhood for variable nodes
	H = np.swapaxes(H,0,1)
	for row_ind, row in enumerate(H):
		neighbour_list = []
		for col_ind, col in enumerate(row):
			if col == 1:
				neighbour_list.append(col_ind)
		M_dict[row_ind] = neighbour_list
	
	return (N_dict, M_dict)

def calc_non_zero(H: np.ndarray) -> Tuple[int, int]:
	"""Returns a list of tuples being the positions in the matrix which aren't zero."""
	non_zero_list = []
	for row_ind, row in enumerate(H):
		for col_ind, col in enumerate(row):
			if col == 1:
				non_zero_list.append((col_ind, row_ind))

	return non_zero_list

def minSum(G: np.ndarray, H: np.ndarray, r: List[int], prob: float, debug_mode = False) -> List[int]:
	""" Min sum algorithm algorithm for AWGN
		Only difference to Gallager is a different function for check node computation
		G: matrix G of size nxk
		H: parity check matrix of size nxh
		r: received message
		N0: channel noise
		(Ec = 1)
	"""

	# Length of message
	n = len(r)

	# Calculate neighbourhood
	N_dict, M_dict = calc_neighbourhood(H=H)

	# List of non_empty entries in H
	non_empty_H_j_i = calc_non_zero(H=H)

	# Initialisation
	L_j = dict()
	L_j_i = dict()
	L_i_j = dict()
	vhat = dict()
	L_j_tot = dict()
	for j in range(n):
		# BSC
		L_j[j] = (-1) ** r[j] * math.log(prob/(1-prob))

		for i in M_dict[j]:
			L_j_i[(j, i)] = L_j[j]
	
	codeword = False
	if debug_mode:
		print("Initial variable nodes (L_v-matrix; index: (x,y)=(width,height)): ")
		print(L_j_i)
	counter = 0
	while not codeword:
		counter += 1
		# Check node update
		# For every entry in H which is not zero
		# for (i,j)
		for (j, i) in non_empty_H_j_i:
			# Compute L_i_j
			L_i_j[(i, j)] = compute_checknode_minsum(N = N_dict[i], L_j_i = L_j_i, i=i, j=j, debug_mode=debug_mode)
		
		if debug_mode:
			print("Check nodes (L_c-matrix; index: (x,y)=(height,width) (index swapped)): ")
			print(L_i_j)
		# Variable node update
		for (j, i) in non_empty_H_j_i:
			# Compute L_j_tot, vhat_j and L_j_i
			L_j_tot[j] = compute_l_j_tot(M = M_dict[j], L_i_j = L_i_j, j = j, Lj = L_j[j], debug_mode=debug_mode)
			vhat[j] = (np.sign(L_j_tot[j])+1) / 2
			L_j_i[(j, i)] = compute_l_j_i(M = M_dict[j], L_i_j = L_i_j, i = i, j = j, Lj = L_j[j])
		
		if debug_mode:
			print("Variable nodes (L_v-matrix; index: (x,y)=(width,height)): ")
			print(L_j_i)
			print("L_j_tot: ")
			print(np.array([L_j_tot[key] for key in sorted(L_j_tot.keys())]))
		# Check termination condition
		vhat_list = np.array([vhat[key] for key in sorted(vhat.keys())])
		# vhat = (vhat_1 ... vhat_n)
		mult = H.dot(vhat_list)%2
		if np.sum(mult) == 0 or counter > 20:
			codeword = True

	"""G_ = np.identity(len(G))
	G2_ = np.zeros((len(G[0]) - len(G), len(G)))
	G_ = np.concatenate((G_, G2_))

	mhat = vhat_list.dot(G_)

	return mhat"""
	return vhat_list

def compute_checknode_minsum(N: List[int], L_j_i, i: int, j: int, debug_mode = False):
	""" Min sum variation of the checknode algorithm
		Does not need any tanh or log computations
		(well, log wasn't necessary for Gallager either) """
	temp = (-1) ** len(N)

	mult = 1
	list_val = []
	str_signs = ""
	for j2 in N:
		if j2 != j:
			mult *= np.sign(L_j_i[(j2, i)])
			str_signs += "*"+str(np.sign(L_j_i[(j2, i)]))
			list_val.append(abs(L_j_i[(j2, i)]))

	if debug_mode:
		print("Checknode minsum (L_i_j, format: (-1)^|N(i)|*min(j'\in N(i)\j)*product over signs of these neighbours): ")
		print("L_("+str(i)+"->"+str(j)+") = -1^"+str(len(N))+"*min("+str(list_val)+str(str_signs)+" = "+str(temp * min(list_val) * mult))

	return temp * min(list_val) * mult

def mldecoder(G, r, comb):
	vhat_list = []
	
	for mhat_poss in comb:
		v_hat = np.array(mhat_poss).dot(G)%2
		vhat_list.append(v_hat)

	best_vhat_min = 10 ** 10
	for vhat_try in vhat_list:
		sum = 0
		for i in range(len(r)):
			if vhat_try[i] == 1:
				sum += (1-r[i]) ** 2
			else:
				sum += (-1-r[i]) ** 2

		if sum < best_vhat_min:
			best_vhat_min = sum
			best_vhat = vhat_try
	
	"""G_ = np.identity(len(G))
	G2_ = np.zeros((len(G[0]) - len(G), len(G)))
	G_ = np.concatenate((G_, G2_))

	mhat = best_vhat.dot(G_)

	return mhat"""
	return best_vhat

# Usage
# python LDPCdecoderBSC.py systematic paritycheck sequence probability debug_mode
# python LDPCdecoderBSC.py G H r prob debug_mode
# python LDPCdecoderBSC.py G.csv H_bec_lecturenotes.csv 011111 0.25 False
def main():
	G = np.loadtxt(sys.argv[1], delimiter=",", dtype=float)
	H = np.loadtxt(sys.argv[2], delimiter=",", dtype=float)
	seq = sys.argv[3]
	prob = float(sys.argv[4])
	dm_str = sys.argv[5]
	if dm_str == "True":
		debug_mode = True
	else:
		debug_mode = False

	# Globally make a list of all possible binary combinations of size k (2^k)
	height_G = len(G)
	comb = [i for i in range(int(math.pow(2, height_G)))]
	comb_list = []
	for i in comb:
		comb_list.append([int(i) for i in np.binary_repr(i, height_G)])
	
	# Encoding
	# Parameters
	(k,n) = np.shape(G)
	R = k/n # Rate of the code

	# maybe adding noise
	# if wanting to encode before
	# m = np.array([1, 0, 1, 0, 1, 1])
	# v = np.matmul(m,G)%2
	# r = v

	# To decode
	r = list(map(int, list(seq)))
	# Task 4
	# r = np.array([-0.2, 0.3, -1.2, 0.5, -0.8, -0.6, 1.1])
	# Example lecture notes page 96
	# r = np.array([-0.9, 0.3, 1.2, 0.9, 0.2, 0.4])
	# N0 = 4

	# Init
	print("Mode: BSC (prob = "+str(prob)+")")
	print("Sequence: "+str(r))

	# Gallager decoding
	print("===========\nCalculate gallager")
	vhat__gallager = gallager(G, H, r, prob, debug_mode=debug_mode)
	print("vhat gallager")
	print(vhat__gallager)
	
	# MinSum decoding
	print("===========\nCalculate minsum")
	vhat_minsum = minSum(G, H, r, prob, debug_mode=debug_mode)
	print("vhat minsum")
	print(vhat_minsum)

	# Maximum-likelihood decoding
	"""print("===========\nCalculate mldecoder")
	mhat_mldec = mldecoder(G, r, comb_list)
	print("mhat max likelihood")
	print(mhat_mldec)"""

if __name__ == "__main__":
	main()