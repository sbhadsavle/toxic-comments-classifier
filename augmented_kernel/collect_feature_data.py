import pandas as pd, numpy as np
import re, string

import perspective
import azure

print("Reading input csv...")
train = pd.read_csv('../../input/train500.csv')

print("Azure text analysis...")
# print(train["comment_text"])
azure_result_df = azure.analyze_text(train["comment_text"])
# print(result_df)

train = train.join(azure_result_df)
# print(train.head())

print("Perspective text analysis...")
perspective_vals = []
for comment in train["comment_text"]:
	if (len(comment) < 3000):
		perspective_vals.append(perspective.get_perspective_toxicity(comment))
	else:
		perspective_vals.append(0.5) # default to "unknown" toxicity if length is too long

persp_df = pd.DataFrame({'perspective_toxicities': perspective_vals})
train = train.join(persp_df)

print("Writing csv...")
train.to_csv('train_augmented.csv', index=False)
print("Done.")
