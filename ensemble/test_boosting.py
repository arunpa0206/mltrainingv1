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

#AdaBoost - Adaptive boosting
from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier(random_state=1)
model.fit(x_train, y_train)
print('Ada Boost accuracy',model.score(x_test,y_test))

#Gradient Bioiosting
from sklearn.ensemble import GradientBoostingClassifier
model= GradientBoostingClassifier(learning_rate=0.01,random_state=1)
model.fit(x_train, y_train)
print('Gradient Boost Accuracy:',model.score(x_test,y_test))

#Extreme Gradient Boosting has pruning and regularization
import xgboost as xgb
model=xgb.XGBClassifier()
model.fit(x_train, y_train)
print('XGBoost:',model.score(x_test,y_test))
