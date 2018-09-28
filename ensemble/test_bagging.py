from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
#import numpy as np
from statistics import mode

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


cancer = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)

#Bagging where the items are bagged as ordered subsets
from sklearn.ensemble import BaggingClassifier
model = BaggingClassifier(DecisionTreeClassifier(random_state=1))
model.fit(x_train, y_train)
print('Bagging score', model.score(x_test,y_test))

#Random bags and average voring is done in Random forest
from sklearn.ensemble import RandomForestClassifier
model= RandomForestClassifier(random_state=1)
model.fit(x_train, y_train)
print('Random forest score:', model.score(x_test,y_test))
