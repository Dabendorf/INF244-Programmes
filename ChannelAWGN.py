#!/usr/bin/env python3

from typing import List
import random
import sys
from PIL import Image
import pandas as pd
import numpy as np

# Random seed of 10 for reproducibility
random.seed(10)

def awgn_img(input: List[int], n_0: int, e_b: int) -> List[int]:
	""" Takes as input an image represented as a list of pixels,
	n_0 representing the standard deviation and e_b being the energy level"""
	# We say that X is normally distributed with mean mu and variance sigma 2 .
	# In symbols, X ~ N ( mu, sigma^ 2 ) .
	# R i = S i + N i ,
	# where N i ~ N ( 0, N 0 /2 )
	# N ( 0, sigma^ 2 )

	# These two lines construct a normal distribution using the given params
	mu, sigma = 0, (n_0/2) # mean and standard deviation
	m_i = np.random.normal(mu, sigma, len(input))
	
	temp = []
	output = []
	# The for-loop is similar to those from BSC and BEC with the different that
	# the output is dependent from the original input.
	# If its lower than 255/2, we calculate the value by using the negative -e_b value,
	# it is bigger we use the positve one since we have two Gauss curves in here
	for i in range(0,len(input)):
		if input[i] < 255/2:
			new_value = -e_b + m_i[i]
		else:
			new_value = +e_b + m_i[i]
		temp.append(new_value)
	temp_min = min(temp)
	temp_max = max(temp)

	# After setting all the new values, there is a need for min-max-normalisation
	# such that the values get spreaded from 0 to 255 to show a proper picture
	for i in range(0,len(input)):
		z_i = 0 + ((temp[i] - temp_min) * (255-0)) / (temp_max - temp_min)
		output.append(z_i)

	return output

# Use: 
# 0.5 = n_0 (sigma), 0.3 = energy level
# python ChannelAWGN.py uib.png 0.5 0.3 png
# python ChannelAWGN.py Mondrian.csv 0.5 0.3 csv
def main():
	# Main method, reads in the file into a Pandas dataframe,
	# reformats it to an numpy array
	# and converts it to an image by scaling it to [0, 255] (which effectively gets 0 or 255)
	# since only 0 or 1 where given
	input_type = sys.argv[4]

	if input_type == "png":
		# This line can substitute the above ones if input is a png file instead of csv
		im1 = Image.open(sys.argv[1]).convert("L")
	else:
		df = pd.read_csv(sys.argv[1], sep=',',header=None, dtype=int)
		df_np = df.to_numpy()
		df_np = (df_np * 255).astype(np.uint8)
		im1 = Image.fromarray(df_np, 'L')

	# Gets pixel data and runs awgn method
	# Awgn method takes two more input params from the consule, being n_0 and e_b
	pixels = list(im1.getdata())
	new_pixels = awgn_img(input = pixels, n_0 = float(sys.argv[2]), e_b = float(sys.argv[3]))
	
	# Writes pixel data back and shows the image
	im1.putdata(new_pixels)
	im1.show()

if __name__ == "__main__":
	main()
