# Baseline code without using any special features.

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier

data_X = []
data_Y = []
with open('processedDataWithoutID.txt') as f:
    for line in f:
        line = line.strip('\n').split('\t')
        data_Y.append(line[1])
        data_X.append(line[0])

train = int(len(data_X)*0.8)

train_X = data_X[:train]
train_Y = data_Y[:train]


test_X = data_X[train:] 
test_Y = data_Y[train:]

test = len(data_X)-train
print train, test
print len(data_X), len(data_Y)
print len(train_X), len(train_Y)
print len(test_X), len(test_Y)


target_names = ['CAG', 'NAG', 'OAG']

# NB
text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),('clf', MultinomialNB()),])

text_clf = text_clf.fit(train_X, train_Y)

predicted = text_clf.predict(test_X)
print np.mean(predicted == test_Y)
# 0.56

print(classification_report(test_Y, predicted, target_names=target_names))

#              precision    recall  f1-score   support

#         CAG       0.51      0.72      0.60       974
#         NAG       0.98      0.13      0.23       466
#         OAG       0.60      0.60      0.60       960

# avg / total       0.64      0.56      0.53      2400


# SVM
text_clf_svm = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf-svm', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42)),  ])

text_clf_svm = text_clf_svm.fit(train_X, train_Y)

predicted_svm = text_clf_svm.predict(test_X)
print np.mean(predicted_svm == test_Y)
# 0.57

print(classification_report(test_Y, predicted_svm, target_names=target_names))


#              precision    recall  f1-score   support

#         CAG       0.54      0.68      0.60       974
#         NAG       0.70      0.31      0.43       466
#         OAG       0.60      0.59      0.59       960

# avg / total       0.59      0.57      0.56      2400


# Decision Tree
text_clf_dt = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),('clf-dt', DecisionTreeClassifier()), ])
text_clf_dt = text_clf_dt.fit(train_X, train_Y)

predict_dt = text_clf_dt.predict(test_X)
print np.mean(predict_dt==test_Y)
# 0.49
print(classification_report(test_Y, predict_dt, target_names=target_names))

#             precision    recall  f1-score   support

#         CAG       0.49      0.50      0.50       974
#         NAG       0.44      0.42      0.43       466
#         OAG       0.53      0.53      0.53       960

# avg / total       0.50      0.50      0.50      2400
