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

train = pd.read_csv('../../input/train_augmented.csv') #.sample(1000)
test = pd.read_csv('../../input/test.csv') #.sample(1000)
subm = pd.read_csv('../../input/sample_submission.csv')

train, val = train_test_split(train, test_size=0.2, random_state=42)
print(train.shape)
print(val.shape)

label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
train['none'] = 1-train[label_cols].max(axis=1)

## take care of empty comments
COMMENT = 'comment_text'
train[COMMENT].fillna("unknown", inplace=True)
val[COMMENT].fillna("unknown", inplace=True)
test[COMMENT].fillna("unknown", inplace=True)

re_tok = re.compile('([' + string.punctuation + '“”¨«»®´·º½¾¿¡§£₤‘’])')

def tokenize(s): 
    return re_tok.sub(r' \1 ', s).split()

def pr(y_i, y):
    p = x[y==y_i].sum(0)
    return (p+1) / ((y==y_i).sum()+1)

def get_mdl(y):
    y = y.values
    r = np.log(pr(1,y) / pr(0,y))
    # m = LogisticRegression(C=4, dual=True)
    m = RandomForestClassifier()
    x_nb = x.multiply(r)
    print("    fitting...")
    return m.fit(x_nb, y), r

print("Computing sparse matrix...")
n = train.shape[0]
vec = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,
               min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
               smooth_idf=1, sublinear_tf=1 )
trn_term_doc = vec.fit_transform(train[COMMENT])
val_term_doc = vec.transform(val[COMMENT])
test_term_doc = vec.transform(test[COMMENT])

# print(trn_term_doc)
# print(test_term_doc)

x = trn_term_doc
x_val = val_term_doc
test_x = test_term_doc

print("val: " + str(x_val.shape[0]) + ", " + str(len(val)))
print(x.shape)
print(x_val.shape)

preds = np.zeros((len(test), len(label_cols)))

print("Starting training...")
for i, j in enumerate(label_cols):
    print('fit', j)
    m,r = get_mdl(train[j])
    preds[:,i] = m.predict_proba(test_x.multiply(r))[:,1]
    print("Accuracy for " + j + ": " + str(accuracy_score(val[j], m.predict(x_val))))

print("Writing csv...")
submid = pd.DataFrame({'id': subm["id"]})
submission = pd.concat([submid, pd.DataFrame(preds, columns = label_cols)], axis=1)
submission.to_csv('submission.csv', index=False)

print("Done.")
