import numpy as np
from sklearn import preprocessing,model_selection as cross_validation, neighbors
import pandas as pd
#Read the creast cancer data from the data file
df = pd.read_csv('breast-cancer-wisconsin.data')
#Clean bad data
df.replace('?',-99999, inplace=True)
#Drop unwanted columns like id
df.drop(['id'], 1, inplace=True)

#Class needs to be predicted. Remove this from the data frame
X = np.array(df.drop(['class'], 1))
#Class is what needs to be predicted
y = np.array(df['class'])
#Use 20% data for testing and 80% data for training
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
#Nearest neighbor classifier from skikit learn
clf = neighbors.KNeighborsClassifier()


clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)
#Pass a new measurement and predict
example_measures = np.array([4,2,1,1,1,2,3,2,1]).reshape(1, -1)
prediction = clf.predict(example_measures)
print(prediction)
