"""Classifier for Toxic Comments.
Authors: Sarang Bhadsavle and Ben Fu
"""

import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split
import re, string

# persistence
from sklearn.externals import joblib

import argparse

from collect_feature_data import collect_features

def tokenize(s):
    re_tok = re.compile('([' + string.punctuation + '“”¨«»®´·º½¾¿¡§£₤‘’])')
    return re_tok.sub(r' \1 ', s).split()

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", nargs=1, default=["web_output.csv"])

args = parser.parse_args()

input_fname = args.file[0]

input_df = pd.read_csv(input_fname)

label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

## take care of empty comments
COMMENT = 'comment_text'
input_df[COMMENT].fillna("unknown", inplace=True)

print(input_df)

# load vectorizer from disk
vec = joblib.load("vectorizer.pkl")
x = vec.transform(input_df[COMMENT])

print(x)

# Now add the additional features
# augmented_features = collect_features(input_df)
x = pd.DataFrame(x.toarray()).fillna(0)
x = input_df[["azure_sentiments", "perspective_toxicities"]].join(x)

print(x)

# load the models from disk
persisted_models = list(map(lambda m: joblib.load(m + ".pkl"), label_cols))
# print(persisted_models)

predicted_probas = np.zeros((len(x), len(label_cols)))

for i, clf in enumerate(persisted_models):
    proba = clf.predict_proba(x)
    predicted_probas[:,i] = proba[:,1]
    pred = clf.predict(x)
    print("Probability for", label_cols[i], ":", proba[0][1])

predicted_probas = input_df[[COMMENT]].join(pd.DataFrame(predicted_probas, columns=label_cols))
print(predicted_probas)