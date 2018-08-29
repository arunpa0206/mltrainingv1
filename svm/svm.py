import numpy as np
from sklearn import preprocessing, model_selection as cross_validation, neighbors, svm
import pandas as pd

#Read data, clean data and drop unwanted features
df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?',-99999, inplace=True)
df.drop(['id'], 1, inplace=True)
#Drop prediction class from data set, Add to y
X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])
#80-20 for train:test
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

clf = svm.SVC()
#Run SVM and calculate confidence
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)
print(confidence)

example_measures = np.array([[4,2,1,1,1,2,3,2,1]])
#-1 in reshape means we want numpy to figure out this dimension
example_measures = example_measures.reshape(len(example_measures), -1)
print(example_measures.shape)
prediction = clf.predict(example_measures)
print(prediction)
