"""Classifier for Toxic Comments.
Authors: Sarang Bhadsavle and Ben Fu
Adapted from original author: Jeremy Howard (https://www.kaggle.com/jhoward/nb-svm-strong-linear-baseline/notebook)
"""

import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split
import re, string

from scipy.sparse import hstack

from collect_feature_data import collect_features

# persistence
from sklearn.externals import joblib

train = pd.read_csv('../../input/train_augmented.csv')# .sample(10)
test = pd.read_csv("web_output.csv")
subm = pd.read_csv('../../input/sample_submission.csv')

# train = collect_features(train)
# test = collect_features(test)

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

def pr(x, y_i, y):
    p = x[y==y_i].sum(0)
    return (p+1) / ((y==y_i).sum()+1)

def get_mdl(x, y):
    y = y.values
    # r = np.log(pr(x, 1,y) / pr(x, 0,y))
    # m = LogisticRegression(C=4, dual=True)
    m = RandomForestClassifier()
    # x_nb = x.multiply(r)
    print("    fitting...")
    return m.fit(x, y)# , r

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
# Save vectorizer
with open("vectorizer.pkl", "wb") as f:
    joblib.dump(vec, f)

# x_train = train[["azure_sentiments", "perspective_toxicities"]].join(pd.DataFrame(trn_term_doc.toarray()).fillna(0, inplace=True))
x_train = hstack((trn_term_doc,np.array(train['azure_sentiments'])[:,None]))
x_train = hstack((x_train,np.array(train['perspective_toxicities'])[:,None]))

# x_val = val[["azure_sentiments", "perspective_toxicities"]].join(pd.DataFrame(val_term_doc.toarray()).fillna(0, inplace=True))
x_val = hstack((val_term_doc,np.array(val['azure_sentiments'])[:,None]))
x_val = hstack((x_val,np.array(val['perspective_toxicities'])[:,None]))

# x_test = test[["azure_sentiments", "perspective_toxicities"]].join(pd.DataFrame(test_term_doc.toarray()).fillna(0, inplace=True))
x_test = hstack((test_term_doc,np.array(test['azure_sentiments'])[:,None]))
x_test = hstack((x_test,np.array(test['perspective_toxicities'])[:,None]))

print(x_train.shape)
print(x_val.shape)

# print(x)

preds = np.zeros((len(test), len(label_cols)))

print("Starting training...")
for i, j in enumerate(label_cols):
    print('fit', j)
    # clf = RandomForestClassifier().fit(x, train[j])
    clf = get_mdl(x_train, train[j])
    # preds[:,i] = m.predict_proba(x_test.multiply(r))[:,1]
    preds[:,i] = clf.predict_proba(x_test)[:,1]
    print("Accuracy for " + j + ": " + str(accuracy_score(val[j], clf.predict(x_val))))

    print("Probability for", j, ":", preds[:,i])

    print("Saving model to disk:", j)
    clf_fname = j + ".pkl"
    with open(clf_fname, "wb") as f:
        joblib.dump(clf, f)

print("Writing csv...")
submid = pd.DataFrame({'id': subm["id"]})
submission = pd.concat([submid, pd.DataFrame(preds, columns = label_cols)], axis=1)
submission.to_csv('submission.csv', index=False)

print("Done.")
