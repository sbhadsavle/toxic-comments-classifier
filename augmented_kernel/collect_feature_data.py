import pandas as pd, numpy as np
import re, string
import time

import sys

import perspective
import azure

import argparse

def azure_text_analysis(train_df):
	# print(train_df["comment_text"])
	result_df = azure.analyze_text(train_df["comment_text"])
	# print(result_df)
	return result_df

def do_get_toxicity(comment):
	toxicity = None
	num_tries = 0
	while True:
		try:
			num_tries += 1
			toxicity = perspective.get_perspective_toxicity(comment)
		except Exception as e:
			if (num_tries < 5):
				print("Perspective API call failed, waiting 100s...")
				time.sleep(100)
				print("...trying again")
				continue
			else:
				# Just go ahead and default comment to 0.5 and move on
				toxicity = 0.5
		break
	return toxicity


def perspective_text_analysis(train_df):
	perspective_vals = []
	for i,comment in enumerate(train_df["comment_text"]):
		print("[Perspective API] Analyzing comment ", i)
		if (len(comment.encode('utf8')) < 3000):
			toxicity = do_get_toxicity(comment)
			perspective_vals.append(toxicity)
		else:
			perspective_vals.append(0.5) # default to "unknown" toxicity if length is too long

	result_df = pd.DataFrame({'perspective_toxicities': perspective_vals})
	return result_df

def collect_features(train):
	print("Azure text analysis...")
	azure_df = azure_text_analysis(train)
	train = train.join(azure_df)
	# print(train.head())

	print("Perspective text analysis...")
	persp_df = perspective_text_analysis(train)
	train = train.join(persp_df)

	return train

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("-f", "--file", nargs=1, default=["../../input/train.csv"])
	parser.add_argument("-o", "--output-file", nargs=1, default=["./train_augmented.csv"])

	args = parser.parse_args()
	input_fname = args.file[0]
	output_fname = args.output_file[0]
	print(input_fname)

	print("Reading input csv...")
	train = pd.read_csv(input_fname)

	df = collect_features(train)

	print("Writing csv...")
	df.to_csv(output_fname, index=False)
	print("Done.")

if __name__ == '__main__':
	main()
