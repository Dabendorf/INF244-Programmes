# Libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from functools import reduce
from typing import List
import sys
import pandas as pd
import math
from collections import defaultdict
import textwrap
import pprint


class ConvEncoder:
	""" Python class doing all the convolutional encoder stuff
	    The methods are inspired by my solution of task 6 from the mandatory assignment 1
		It is different version than task 3, since it has no feedback, but bigger matrixes"""
	
	def __init__(self, k: int, n: int, b, biggest_mem: int, mem_sum: int, mem_col, r: float) -> None:
		self.k = k
		self.n = n
		self.b = b
		self.biggest_mem = biggest_mem
		self.mem_sum = mem_sum
		self.mem_col = mem_col
		self.r = r

		self.memory = 0

		self.trans_matrix = {} # Dictionary with transition information
		self.next_state_matrix = {}

		# Generates list of all possible input combinations
		self.input_kombinations = [f'{i:0{k}b}' for i in range(int(math.pow(2, k)))]

		# Generates a list of all possible states
		self.big_state_list = [f'{i:0{self.mem_sum}b}' for i in range(int(math.pow(2, self.mem_sum)))]

		for state in self.big_state_list:
			for bit_comb in self.input_kombinations:
				# Loop over amount of rows
				offset = 0
				ind_counter = 0
				mem_overall = ""
				for mem_el in self.mem_col:
					new_mem = bit_comb[ind_counter] + state[offset: offset+mem_el][:-1]
					offset += mem_el
					ind_counter += 1
					mem_overall += new_mem
				
				self.next_state_matrix[(state, bit_comb)] = mem_overall

		# Transition matrix
		# Two outer for-loops to loop through all possible state and input combinations
		for state in self.big_state_list:
			for bit_comb in self.input_kombinations:
				counter = 0
				output_str = ""
				# It then loops through the columns and the rows
				# It makes an AND operation over the combination of input and memory and the polynomial of one m
				# The bits then get XORt all over which gives the pre-output for one cell of the matrix
				# Those then get XORt over all rows
				# The data achieved from here gets into the dictionary and the algorithm is to generate
				# the transition matrix is finished
				for column in b:
					current_bit = 0
					row_counter = 0
					row_index_counter = 0
					for row in column:
						memory = bit_comb[row_counter] + state[row_index_counter:row_index_counter+mem_col[row_counter]]
						row_index_counter += mem_col[row_counter]
						diff = len(row) - len(memory)
						if diff > 0:
							memory = memory + diff*"0"
						result = int(memory, 2) & int(row, 2)
						temp_bit = multi_xor(result)
						current_bit ^= temp_bit
						row_counter += 1
					
					output_str += str(current_bit)
					# print("c="+str(counter)+", "+str(current_bit))
					counter += 1
				self.trans_matrix[(state, bit_comb)] = output_str

	def encode(self, seq: str) -> List[int]:
		# Encoding stuff
		#seq = np.array(seq)
		#chunks = np.array_split(seq, len(seq)/self.k)
		print("Memory length "+str(self.mem_sum)+", therefore added bits")
		seq += "0" * (self.k * self.mem_sum)
		print("New input message (length="+str(len(seq))+")")
		print(seq)

		chunks = textwrap.wrap(seq, self.k)
		self.memory = "0" * self.mem_sum
		
		encoded_string = ""

		for chunk in chunks:
			#chunk_str = ''.join(str(x) for x in chunk)
			chunk_str = chunk
			encoded_bits = self.trans_matrix[(self.memory, chunk_str)]
			encoded_string += encoded_bits
			self.memory = self.next_state_matrix[(self.memory, chunk_str)]

		print("Memory is zerofied: "+str(int(self.memory)==0))

		return encoded_string
		


def multi_xor(a: int) -> int:
	""" Performs xor over all bits of a number 
		Taken from my solution from mandatory assignment 1"""
	temp = a
	val = 0
	while temp != 0:
		val ^= (temp%2)
		temp //= 2
	
	return val

# Use: 
# Input: Size of matrix, matrix itself, to_decode
# python ConvFinal.py k n matrix seq
# Examples: 
# python ConvFinal.py 1 2 111S101 100100
# python ConvFinal.py 1 2 1111S1011 1011011
def main():
	# START - Ugly Code from Assignment 1
	k = int(sys.argv[1])
	n = int(sys.argv[2])
	rate = k/n
	b_str = sys.argv[3]
	seq = sys.argv[4]
	
	# Reads the matrix in the same way as it did in the other programmes
	# Has S as separator symbol
	b_temp = b_str.split("S")
	b_temp_str_bin = []
	max_size = 0
	for i in b_temp:
		if len(i) > max_size:
			max_size = len(i)
		b_temp_str_bin.append(i)
	
	# Fills with zeros at the end such that all strings are equally long
	b_temp_bin = []
	for i in b_temp_str_bin:
		diff = max_size - len(i)
		if diff > 0:
			a = i + diff*"0"
		else:
			a = i
		b_temp_bin.append(a)

	
	b = np.array(b_temp_bin).reshape(k, n)
	b_old = b
	b = np.swapaxes(b, 0, 1)

	# Finds the sizes of all memory cells
	v = max_size-1
	mem_col = []
	for col in b_old:
		highest_power = 0
		for el in col:
			counter = 0
			for cha in el:
				if cha == "1" and counter > highest_power:
					highest_power = counter
				counter += 1
		mem_col.append(highest_power)

	mem_col_sum = sum(mem_col)

	# Variables:
	# k and n: size of matrix
	# b: Generator matrix
	# v: highest memory amount on a row
	# mem_col_sum: Constraint length
	# rate: k/n

	conv = ConvEncoder(k = k, n= n, b = b, biggest_mem = v, mem_sum = mem_col_sum, mem_col = mem_col, r = rate)

	print("Next state matrix (mem, input)->mem")
	print(pprint.pprint(conv.next_state_matrix, width=1))
	print("Transition matrix (mem, input)->output")
	print(pprint.pprint(conv.trans_matrix, width=1))

	print("Original sequence (length="+str(len(seq))+")")
	print(seq)

	encoded_seq = conv.encode(seq)

	print("Encoded sequence (length="+str(len(encoded_seq))+")")
	print(encoded_seq)
	print("Length check: "+str(((len(seq)/k)*n)+(mem_col_sum*n)==len(encoded_seq)))


if __name__ == "__main__":
	main()