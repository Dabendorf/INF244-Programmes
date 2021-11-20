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

class Viterbi:
	""" Viterbi programme trying to decode a message
		given the state and the transition matrix of a convolutional encoder"""

	def __init__(self, next_state_matrix, trans_matrix, r: List[int], big_state_list, n: int, channel_mode) -> None:
		# Input of Viterbi:
		# Sequence r to decode
		# Next_state_matrix and transition_matrix (Output bits)
		# State-Space: All possible states (s / big_state_list)
		# Predecessors: P(s \in S) All states that can turn into s by single input
		self.next_state_matrix = next_state_matrix
		self.trans_matrix = trans_matrix
		self.r = r
		self.s = big_state_list
		self.n = n # Amount of output bits
		self.size_mem = len(self.s[0])

		# Now, convert the state matrix into easier ones, ignoring the input bits
		# New:
		# next_state_easy[memory] = [new_memory_possible]
		self.next_state_easy = defaultdict(list)
		for key, value in next_state_matrix.items():
			self.next_state_easy[key[0]].append(value)

		# Calculates invert of transition matrix, precedessor dictionary
		# p is not just the predecessor, but a tuple (predecessor, input)
		self.p = defaultdict(list)
		for trans_key, trans_value in self.next_state_matrix.items():
			self.p[trans_value].append(trans_key[0])

		# Length of sequence
		if channel_mode=="awgn":
			awgn_str = self.r
			awgn_list_temp = awgn_str.split("S")
			self.r = list(map(float, awgn_list_temp))
			print("To decode: "+str(self.r))
			print("Length: "+str(len(self.r)))
		else:
			print("To decode: "+str(self.r))
			print("Length: "+str(len(self.r)))
		max_depth = (len(self.r)//n)

		# Calculate dict which prints output bits depending on state transition
		get_output_bits = dict()
		for key, value in self.next_state_matrix.items():
			get_output_bits[(key[0], value)] = self.trans_matrix[(key[0], key[1])]

		# calculate dict which prints input bits depending on state transition
		get_input_bits = dict()
		for key, value in self.next_state_matrix.items():
			get_input_bits[(key[0], value)] = key[1]

		# Calculate S_d
		# States reachable at depth d
		# Format of s_d:
		# s[depth] = ['00', '01']
		
		s_d = defaultdict(list)
		s_d[-1] = [self.size_mem*"0"]

		for ind in range(max_depth):
			for st in s_d[ind-1]:
				states_next_round = self.next_state_easy[st]
				for snr in states_next_round:
					if snr not in s_d[ind]:
						s_d[ind].append(snr)

		# Problem left:
		# Since the programme should end in '00',
		# s_d should only include paths ending in 00 at the end
		s_d[max_depth-1] = [self.size_mem*"0"]
		for ind in range(max_depth-2, 1, -1):
			# Break loop if state afterwards has all states available
			if len(s_d[ind+1]) == len(self.s):
				break
			else:
				remove_list = []
				for state in s_d[ind]:
					for int_state in self.next_state_easy[state]:
						if int_state in s_d[ind+1]:
							break
						remove_list.append(state)
				s_d[ind] = [klinger for klinger in s_d[ind] if klinger not in remove_list]

		# Calculate P_d
		# Predecessor states at depth d
		# Based on s_d and p
		p_d = defaultdict(lambda: defaultdict(list))
		for ind in range(1, len(s_d)):
			for state in s_d[ind]:
				p_d[ind][state].extend(self.p[state])

		# Same as for s_d, we need to remove non-available paths, now from the beginning
		for state in s_d[0]:
			p_d[0][state] = [self.size_mem*"0"]
		for ind in range(1, max_depth-2):
			# Break loop if state afterwards has all states available
			if len(p_d[ind-1]) == len(self.s):
				break
			else:
				remove_list = []
				for int_state in p_d[ind]:
					remove_list = []
					for reachable in p_d[ind][int_state]:
						if reachable not in s_d[ind-1]:
							remove_list.append(reachable)
					p_d[ind][int_state] = [klinger for klinger in p_d[ind][int_state] if klinger not in remove_list]
		
		# Algorithm itself
		d = 0

		# Initialise empty lists in lab_dictionary
		o = dict()
		s_0 = self.s[0] # zero state
		lab = dict()
		for state in self.s:
			lab[state] = []
			o[state] = 0
		lab[s_0] = ""

		# While loop depending on depth (length of sequence)
		while d < max_depth: 
			o2 = dict()
			lab2 = dict()
			for state in s_d[d]:
				v_p = dict()
				for pre in p_d[d][state]:
					# Calc output bits
					c_d = get_output_bits[(pre, state)]
					#if channel_mode != "awgn":
					v_p[pre] = o[pre] + self.error_metric(r = self.r[d*n:(d+1)*n], tested_sequence = c_d, mode=channel_mode)
					#else:
						#v_p[pre] = o[pre] + self.error_metric(r = awgn_list[d*n:(d+1)*n], tested_sequence = c_d, mode=channel_mode)

				p_0 = min(v_p, key = v_p.get)
				lab2[state] = (lab[p_0] + state)
				o2[state] = v_p[p_0]

			for state in s_d[d]:
				o[state] = o2[state]
				lab[state] = lab2[state]

			d += 1

		# Reconstruct path
		# self.size_mem
		self.result = np.array([])
		start = self.size_mem*"0"

		for i in range(max_depth):
			finish = lab[s_0][i*self.size_mem:(i+1)*self.size_mem]

			self.result = np.append(self.result, list(get_input_bits[(start, finish)]))
			start = finish

	def error_metric(self, r, tested_sequence, mode: str) -> int:
		""" Takes two sequences, outputs the error metric result
			BSC: it is just an integer how many bits are wrong
			BEC: 
			AWGN:
		"""
		if mode=="bsc":
			counter = 0
			for i in range(len(r)):
				if str(r[i]) != tested_sequence[i]:
					counter += 1

			return counter
		elif mode=="bec":
			counter = 0
			for i in range(len(r)):
				if str(r[i]) != tested_sequence[i]:
					counter += 100000

			return counter
		elif mode=="awgn":
			print("Error metric")
			print(r)
			print(tested_sequence)
			sum = 0
			for i in range(len(r)):
				sum += (abs(r[i]-float(tested_sequence[i]))**2)
			return sum
		else:
			print("Abortion error: Choose channel from {bsc, bec, awgn}")
			exit()


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
# Input: Size of matrix, channel, matrix itself, encodec_sequence
# python3 Viterbi.py k n channel matrix seq
# Examples: 
# python Viterbi.py 1 2 bsc 111S101 000111011001110000
# python Viterbi.py 1 2 bsc 111S001 000111011001110000
# python Viterbi.py 2 3 bsc 111S11S101S1S01S11 111000011100000100
def main():
	# START - Ugly Code from Assignment 1
	k = int(sys.argv[1])
	n = int(sys.argv[2])
	rate = k/n
	channel = sys.argv[3]
	b_str = sys.argv[4]
	seq = sys.argv[5]
	
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
	print("Convolutional encoder")
	print("Next state matrix (mem, input)->mem")
	print(conv.next_state_matrix)
	print(pprint.pprint(conv.next_state_matrix, width=1))
	print("Transition matrix (mem, input)->output")
	print(pprint.pprint(conv.trans_matrix, width=1))
	# END - Ugly Code from Assignment 1

	# Decoding
	print("Viterbi, mode: "+channel)
	print(seq)
	vit = Viterbi(conv.next_state_matrix, conv.trans_matrix, seq, conv.big_state_list, n, channel)
	mhat = vit.result

	mhat = mhat.astype(np.int)

	# Attention: Since ViteriMC is my original programme and Viterbi.py is just derived from it
	#the output might look a bit weird
	# depending on k, every k-th bit is a bit from the k-th input string
	# Example:
	# For r=1/2: [c_0, c_1, c_2, ...]
	# For r=2/3: [c(0)_0, c(1)_0, c(0)_1, c(1)_1, c(2)_0, ...]
	print("The original message was: ")
	print(mhat)

	


if __name__ == "__main__":
	main()