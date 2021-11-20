#!/usr/bin/env python3

from typing import List
import random
import sys
from PIL import Image
import pandas as pd
import numpy as np

def bsc_img(input: List[int], prob: int) -> List[int]:
	""" Method is similar to that one from task 2"""
	output = []
	for i in input:
		if random.random() < prob:
			new_value = (255-i)
		else:
			new_value = i
		output.append(new_value)

	return output

def bec_img(input: List[int], prob: int) -> List[int]:
	""" Method is similar to that one from task 2"""
	output = []
	for i in input:
		if random.random() < prob:
			new_value = 255/2
		else:
			new_value = i
		output.append(new_value)

	return output

def awgn_img(input: List[int], n_0: int, e_b: int) -> List[int]:
	""" Method is similar to that one from task 2"""
	mu, sigma = 0, (n_0/2) # mean and standard deviation
	m_i = np.random.normal(mu, sigma, len(input))
	
	temp = []
	output = []
	for i in range(0,len(input)):
		if input[i] < 255/2:
			new_value = -e_b + m_i[i]
		else:
			new_value = +e_b + m_i[i]
		temp.append(new_value)
	temp_min = min(temp)
	temp_max = max(temp)

	for i in range(0,len(input)):
		z_i = 0 + ((temp[i] - temp_min) * (255-0)) / (temp_max - temp_min)
		output.append(z_i)

	return output

def rep_code(input: List[int], n: int):
	""" Method takes as input sequence of bits and n and
	outputs sequence with length len(input) * n, with n repeated pixels
	for every original pixel"""
	output = []

	# Just for-loops through all pixel and outputs every pixel n times
	for pix in input:
		for i in range(n):
			output.append(pix)

	return output

def bsc_likelihood(input: List[int], n: int):
	# Weird python function seperating list into chunks of size n
	chunks = [input[i * n:(i + 1) * n] for i in range((len(input) + n - 1) // n )]
	
	output = []

	# Loops through all chunks
	for chunk in chunks:
		# Counts black and white pixels
		count_w = 0
		count_b = 0

		for el in chunk:
			if el > 255/2:
				count_b += 1
			else:
				count_w += 1
		# Returns the pixel appearing more often following the ML rule
		output.append(255 if count_b > count_w else 0)

	return output

def bec_likelihood(input: List[int], n: int):
	chunks = [input[i * n:(i + 1) * n] for i in range((len(input) + n - 1) // n )]
	
	output = []
	for chunk in chunks:
		# Finds out if there is a 0 or a 255 in the list
		# Both of it is not possible
		# Returns erased bit if all bits in chunk are erased
		if 0 in chunk:
			output.append(0)
		elif 255 in chunk:
			output.append(255)
		else:
			output.append(127.5)

	return output

def awgn_likelihood(input: List[int], n: int):
	chunks = [input[i * n:(i + 1) * n] for i in range((len(input) + n - 1) // n )]

	output = []
	# Loops through all chunks
	for chunk in chunks:
		"""# Counts black and white pixels
		wrong
		count_w = 0
		count_b = 0

		for el in chunk:
			if el > 255/2:
				count_b += 1
			else:
				count_w += 1
		# Returns the pixel appearing more often following the ML rule
		# The count was happening by looking for 255/2 as decision boundary,
		# Which is just a shifted version of the 0 zero boundary from the lecture notes.
		output.append(255 if count_b > count_w else 0)"""
		chunksum = np.sum(chunk)
		output.append(255 if chunksum > n*255/2 else 0)

	return output

def calc_changes(list1: List[int], list2: List[int]) -> float:
	# Just counts how many pixel are different in both lists
	# The function expects both lists to be equally sized
	# since the image does not change the amount of pixels
	counter = 0
	for i in range(len(list1)):
		if list1[i] != list2[i]:
			counter += 1
	return counter/len(list1)

# Use
# 5 = repetitions, "bsc" = channelname
# param is the crossover probability for bsc and bec
# or E_b/N_O for AWGN
# python Repcode.py Mondrian.csv 5 channelname param
# python Repcode.py Mondrian.csv 5 bsc 0.3
# python Repcode.py Mondrian.csv 5 bec 0.3
# python Repcode.py Mondrian.csv 5 awgn 0.6
def main():
	# Same input code as for those in task 2
	df = pd.read_csv(sys.argv[1], sep=',',header=None, dtype=int)
	df_np = df.to_numpy()
	df_np = (df_np * 255).astype(np.uint8)
	im1 = Image.fromarray(df_np, 'L')
	pixels = list(im1.getdata())

	# Transforms pixel values into repetition code
	rep_pixel = rep_code(input = pixels, n = int(sys.argv[2]))
	
	# Takes as input name of a channel, bsc, bec or awgn
	channel_str = sys.argv[3]
	par = float(sys.argv[4]) # this is the crossover probability for bsc and bec
	# or E_B/N_0 for AWGN

	# Chooses the channel, runs the image function of the channel
	# and then the likelihood method
	# It then outputs the image again
	if channel_str == "bsc":
		bsc_pixels = bsc_img(input = rep_pixel, prob = par)
		bsc_pixels2 = bsc_likelihood(input = bsc_pixels, n = int(sys.argv[2]))
		
		print("Error rate: "+str(calc_changes(pixels, bsc_pixels2)))

		im1.putdata(bsc_pixels2)
		im1.show()

		im1.save("bscRep.png", format="png")

	elif channel_str == "bec":
		bec_pixels = bec_img(input = rep_pixel, prob = par)
		bec_pixels2 = bec_likelihood(input = bec_pixels, n = int(sys.argv[2]))

		print("Error rate: "+str(calc_changes(pixels, bec_pixels2)))
		
		im1.putdata(bec_pixels2)
		im1.show()

		im1.save("becRep.png", format="png")

	elif channel_str == "awgn":
		# e_b is always 1 (to not make things overcomplicated)
		# and n_0 gets calculate by using the input param E_b/N_0
		e_b = 1
		n_0 = 1/par
		awgn_pixels = awgn_img(input = rep_pixel, n_0 = n_0, e_b = e_b)
		awgn_pixels2 = awgn_likelihood(input = awgn_pixels, n = int(sys.argv[2]))

		print("Error rate: "+str(calc_changes(pixels, awgn_pixels2)))
		
		im1.putdata(awgn_pixels2)
		im1.show()

		im1.save("awgnRep.png", format="png")
	else:
		print("Please use either bsc, bec or awgn as channel name")

if __name__ == "__main__":
	main()