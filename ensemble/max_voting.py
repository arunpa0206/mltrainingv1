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

from sklearn.ensemble import VotingClassifier
model1 = LogisticRegression(random_state=1)
model2 = DecisionTreeClassifier(random_state=1)

model1.fit(x_train,y_train)
print(' model 1 accuracy:',model1.score(x_test,y_test))

model2.fit(x_train,y_train)
print(' model 2 accuracy:',model2.score(x_test,y_test))

model = VotingClassifier(estimators=[('lr', model1), ('dt', model2)], voting='hard')
model.fit(x_train,y_train)
print('Ensemble model accuracy:',model.score(x_test,y_test))
