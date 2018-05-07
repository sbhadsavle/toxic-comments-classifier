"""Classifier for Toxic Comments.
Authors: Sarang Bhadsavle and Ben Fu
"""

import pandas as pd, numpy as np

from sklearn.neural_network import MLPClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout, MaxPooling1D
from keras.optimizers import Nadam
from keras.models import model_from_json

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split
import re, string

from scipy.sparse import hstack

from collect_feature_data import collect_features

# persistence
from sklearn.externals import joblib

# evaluation utilities
import os
import sys
sys.path.append('../evaluation_utils/')
import evaluation_utils

train = pd.read_csv('../../input/train_augmented.csv')#.sample(10000)
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

    m = Sequential()
    opt = Nadam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    m.add(Dense(units=512, init='uniform', activation='tanh', input_dim=x.shape[1]))
    m.add(Dropout(0.15)) 
    m.add(Dense(units=1, init='uniform', activation='relu'))
    m.compile(loss='binary_crossentropy',
              optimizer='Nadam',
              metrics=['accuracy'])
    m.fit(x, y, epochs=3, batch_size=128)

    # m = MLPClassifier()
    # print("    fitting...")
    # m.fit(x, y)

    return m

def roundfunc(x_val):
    if x_val >= 0.5:
        x_val = 1
    else:
        x_val = 0
    return x_val

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

x_train = x_train.tocsr()
x_test = x_test.tocsr()
x_val = x_val.tocsr()

print("Starting training...")
for i, j in enumerate(label_cols):
    print('fit', j)
    clf = None
    if os.path.isfile(str(j) + "_clf.json"):
        json_file = open(str(j) + "_clf.json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        clf = model_from_json(loaded_model_json)
        # load weights into new model
        clf.load_weights(str(j) + "_clf_weights.h5")
        print("Loaded model from disk")
    else:
        clf = get_mdl(x_train, train[j])
    # preds[:,i] = m.predict_proba(x_test.multiply(r))[:,1]
    # preds[:,i] = clf.predict_proba(x_test)[:,1]
    print("Accuracy for " + j + ": " + str(accuracy_score(val[j], [ roundfunc(x) for x in clf.predict(x_val) ])))
    print("Writing reports...")
    reports_dir = "reports/"
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)
    final_preds = [ roundfunc(x) for x in clf.predict(x_val) ]
    evaluation_utils.write_classification_report(final_preds, val[j], "reports/neural_classification_report_" + j + ".txt")
    evaluation_utils.write_confusion_matrix(final_preds, val[j], "Neural Network Performance: " + j, "reports/neural_confusion_matrix_" + j + ".png")
    # print("Probability for", j, ":", preds[:,i])

    # serialize model to JSON
    model_json = clf.to_json()
    with open(str(j) + "_clf.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    clf.save_weights(str(j) + "_clf_weights.h5")
    print("Saved model to disk")

print("Done.")
