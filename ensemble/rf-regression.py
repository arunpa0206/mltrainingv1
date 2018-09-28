from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

X, y = make_regression(n_features=4, n_informative=2,
                        random_state=0, shuffle=False)

print('Sample data')
print(X[0],y[0])


regr = RandomForestRegressor(max_depth=2, random_state=0)
regr.fit(X, y)
print('Prediction:')
print(regr.predict([[0, 0, 0, 0]]))
