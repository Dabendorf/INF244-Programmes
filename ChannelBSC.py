#!/usr/bin/env python3

from typing import List
import random
import sys
from PIL import Image
import pandas as pd
import numpy as np

# Random seed of 10 for reproducibility
random.seed(10)

def bsc_img(input: List[int], prob: int) -> List[int]:
	""" Takes as input an image represented as a list of pixels
	    and a probability \in [0,1]"""
	output = []
	# For loop through all pixel
	# If a random value is smaller than the probability then value gets inverted
	for i in input:
		if random.random() < prob:
			new_value = (255-i)
		else:
			new_value = i
		output.append(new_value)

	return output

# Use: 
# 0.1 = prob
# python ChannelBSC.py uib.png 0.1 png
# python ChannelBSC.py Mondrian.csv 0.1 csv
def main():
	# Main method, reads in the file into a Pandas dataframe,
	# reformats it to an numpy array
	# and converts it to an image by scaling it to [0, 255] (which effectively gets 0 or 255)
	# since only 0 or 1 where given
	input_type = sys.argv[3]

	if input_type == "png":
		# This line can substitute the above ones if input is a png file instead of csv
		im1 = Image.open(sys.argv[1]).convert("L")
	else:
		df = pd.read_csv(sys.argv[1], sep=',',header=None, dtype=int)
		df_np = df.to_numpy()
		df_np = (df_np * 255).astype(np.uint8)
		im1 = Image.fromarray(df_np, 'L')

	# Gets pixel data and runs bsc method
	pixels = list(im1.getdata())
	new_pixels = bsc_img(pixels, float(sys.argv[2]))

	# Writes pixel data back and shows the image
	im1.putdata(new_pixels)
	im1.show()

if __name__ == "__main__":
    main()
