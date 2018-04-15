import pandas as pd, numpy as np
import re, string

import perspective
import azure

def azure_text_analysis(train_df):
	# print(train_df["comment_text"])
	result_df = azure.analyze_text(train_df["comment_text"])
	# print(result_df)
	return result_df

def perspective_text_analysis(train_df):
	perspective_vals = []
	for i,comment in enumerate(train_df["comment_text"]):
		print("[Perspective API] Analyzing comment ", i)
		if (len(comment) < 3000):
			perspective_vals.append(perspective.get_perspective_toxicity(comment))
		else:
			perspective_vals.append(0.5) # default to "unknown" toxicity if length is too long

	result_df = pd.DataFrame({'perspective_toxicities': perspective_vals})
	return result_df

def main():
	print("Reading input csv...")
	train = pd.read_csv('../../input/train500.csv')

	print("Azure text analysis...")
	azure_df = azure_text_analysis(train)
	train = train.join(azure_df)
	# print(train.head())

	print("Perspective text analysis...")
	persp_df = perspective_text_analysis(train)
	train = train.join(persp_df)

	print("Writing csv...")
	train.to_csv('train_augmented.csv', index=False)
	print("Done.")

if __name__ == '__main__':
	main()
