#!/usr/bin/env python3
from functools import reduce
from typing import List
import sys
import numpy as np
import pandas as pd
import math
import pprint

class ConvEncoder:
	""" Python class doing all the convolutional encoder stuff
	    The methods are inspired by my solution of task 6 from the mandatory assignment 1
		The only difference is that the matrix is 1x1 and there is a denominator"""
	
	def __init__(self, num: int, denom: int, input: int, mem_len: int):
		self.num = num
		self.denom = denom
		self.input = input

		self.memory_len = mem_len
		self.memory = 0
		
		self.current_input_index = 0

		self.state_matrix = self.calc_next_state_matrix(mem_len = self.memory_len)

		self.transition_matrix = self.calc_transition_matrix(mem_len = self.memory_len)

		self.output_list = []

		# For loop looping through all input strings checking the matrix dictionaries
		for ind in range(len(input)):
			new_output = self.transition_matrix[(self.memory, self.input[ind])]
			self.output_list.append(new_output)

			self.memory = self.state_matrix[(self.memory, self.input[ind])]

	def next_output(self, current_mem: int, input_bit: int, mem_len: int) -> int:
		""" Calculates the output for one specific input and memory 
			Helper function for calc_transition_matrix """
		# First calculate the feedback
		altered_denom = int(bin(self.denom)[3:],2)
		feedback = multi_xor(current_mem & altered_denom)

		# Now calculate the input, named overall_memory here
		overall_mem = 0
		input_final = input_bit ^ feedback

		# If the new input (XORed with feedback) is 1, concat to the memory
		if input_final == 1:
			overall_mem = 2**mem_len + current_mem
		else:
			overall_mem = current_mem

		output = multi_xor(overall_mem & self.num)
		return output

	def calc_transition_matrix(self, mem_len: int):
		""" Generates the entire transition matrix"""
		output_dict = {}
		state_list = [f'{i:0{mem_len}b}' for i in range(int(math.pow(2, mem_len)))]
		for single_bit in [0, 1]:
			for state in state_list:
				new_output = self.next_output(current_mem = int(state, 2), input_bit = single_bit, mem_len=mem_len)
				output_dict[(int(state,2), single_bit)] = new_output
		
		return output_dict
	
	def next_memory(self, current_mem: int, input_bit: int, mem_len: int) -> int:
		""" Calculates next memory state by bit shifting 
		Helper function for calc_next_state_matrix """
		# First calculate the feedback
		altered_denom = int(bin(self.denom)[3:],2)
		feedback = multi_xor(current_mem & altered_denom)
		input_final = input_bit ^ feedback

		if input_final == 0:
			return current_mem >> 1
		else:
			return 2**mem_len + current_mem >> 1

	def calc_next_state_matrix(self, mem_len: int):
		""" Generates next state matrix, inspired from assignment 1, but easier (since only one input)"""
		next_state_dict = {}
		state_list = [f'{i:0{mem_len}b}' for i in range(int(math.pow(2, mem_len)))]
		for single_bit in [0, 1]:
			for state in state_list:
				new_state = self.next_memory(current_mem = int(state, 2), input_bit = single_bit, mem_len = mem_len)
				next_state_dict[(int(state,2), single_bit)] = new_state

		return next_state_dict




def multi_xor(a: int) -> int:
	""" Performs xor over all bits of a number 
		Taken from my solution from mandatory assignment 1"""
	temp = a
	val = 0
	while temp != 0:
		val ^= (temp%2)
		temp //= 2
	
	return val

def permute(seq: List[int], perm: List[int]):
	""" Permutes the sequence
		Also checks if input is zero or one-based"""
	new_seq = [-1] * len(seq)
	if 0 in perm:
		for i, el in enumerate(perm):
			new_seq[el] = seq[i]
	else:
		for i, el in enumerate(perm):
			new_seq[el-1] = seq[i]

	return new_seq

def conc_three_lists(c0: List[int], c1: List[int], c2: List[int]):
	""" Outputs a new string which concatinates the three c-outputs
		c(0)_1, c(1)_1, c(2)_1, ... , c(0)_n, c(1)_n, c(2)_n
		It doen't check if all lists have equal size,
		but it cannot happen that this is not the case"""
	c = ""
	for i in range(len(c0)):
		c += str(c0[i])
		c += str(c1[i])
		c += str(c2[i])

		c += ", "

	return c[:-2] #removes the last ", " at the end

# Example:
# python TurboEncoder.py g type m permuter
# python TurboEncoder.py 1S1101 csv Sequence.csv Permutation.csv
# python TurboEncoder.py 1S1101 str 10110 4S5S3S1S2

# Which represents
# G(x) as input, having S as separator between numerator and denominator
# m: Message as csv input
# permuter: Permutation as csv input
def main():
	# Reads matrix (1x1)
	g_temp = sys.argv[1]
	num, denom = g_temp.split("S")[0:2] # Splits into numerator and denomenator

	# Reads sequence separated by comma

	input_type = sys.argv[2]

	if input_type == "csv":
		seqDF = pd.read_csv(sys.argv[3], sep=',',header=None, dtype=int)
		m = list(np.array(seqDF).reshape(-1)) # which also is C0

		# Reads sequence separated by newlines (why, Andrea?)
		permuterDF = pd.read_csv(sys.argv[4], sep='\n',header=None, dtype=int)
		permuter = np.array(permuterDF).reshape(-1)
		
	else:
		m = np.array(list(map(int, list(sys.argv[3]))))
		permuter = np.array(list(map(int, sys.argv[4].split("S"))))

	print("Message")
	print(m)
	print("Permuter")
	print(permuter)

	# Fill g_temp with zeros
	diff_g_len = len(num) - len(denom)
	if diff_g_len > 0:
		denom += ("0" * diff_g_len)
	elif diff_g_len < 0:
		num += ("0" * abs(diff_g_len))

	perm = permute(seq = list(m), perm = list(permuter))
	print("Permuted message: "+str(perm))

	# Normal convolutional encoder
	conv1 = ConvEncoder(num=int(num, 2), denom=int(denom, 2), input=m, mem_len=len(num)-1)
	print("Normal convolutional encoder")
	print("Next state matrix (mem, input)->mem")
	print(pprint.pprint(conv1.state_matrix, width=1))
	print("Transition matrix (mem, input)->output")
	print(pprint.pprint(conv1.transition_matrix, width=1))

	# Permuted convolutional encoder
	conv2 = ConvEncoder(num=int(num, 2), denom=int(denom, 2), input=perm, mem_len=len(num)-1)
	print("Permuted convolutional encoder")
	print("Next state matrix (mem, input)->mem")
	print(pprint.pprint(conv2.state_matrix, width=1))
	print("Transition matrix (mem, input)->output")
	print(pprint.pprint(conv2.transition_matrix, width=1))

	c1 = conv1.output_list
	c2 = conv2.output_list

	print("c0 = "+str(m))
	print("c1 = "+str(c1))
	print("c2 = "+str(c2))
	print("c  = "+conc_three_lists(m, c1, c2))


if __name__ == "__main__":
	main()