# You are free to reuse this code anywhere you want
# Created by Tinoxmr
# 25/09/2019
#
# This simple Python script compare the performance of an SVM classifier and a DecisionTree classifier
# using the 'digits' standard dataset from sklearn.datasets


from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn import tree

digits = datasets.load_digits()

n_samples = len(digits.data)

# build SVM classifier
clf_SVC = svm.SVC(gamma=0.001, C=10)
# train classifier with half of the data
clf_SVC.fit(digits.data[:n_samples//2], digits.target[:n_samples//2])
# build Decision Tree classifier
clf_DT = tree.DecisionTreeClassifier()
# train DT classifier with half of the data
clf_DT = clf_DT.fit(digits.data[:n_samples//2], digits.target[:n_samples//2])

# Now predict the value of the digit on the second half using SVC:
expected = digits.target[n_samples//2:]
predicted = clf_SVC.predict(digits.data[n_samples//2:])
# Print report for SVC classifier
print("Classification report for classifier:\n", clf_SVC, "\n", metrics.classification_report(expected, predicted))

# Now predict the value of the digit on the second half using Decision Tree:
expected = digits.target[n_samples//2:]
predicted = clf_DT.predict(digits.data[n_samples//2:])
# Print report for DT classifier
print("Classification report for classifier:\n", clf_DT, "\n", metrics.classification_report(expected, predicted))

# print Decision Tree in text format
r = tree.export.export_text(clf_DT)
print("Decision Tree:")
print(r)
