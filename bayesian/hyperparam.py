import numpy as np
from hyperopt import hp, tpe, fmin

# Single line bayesian optimization of polynomial function
best = fmin(fn = lambda x: np.poly1d([1, -2, -28, 28, 12, -26, 100])(x),
            space = hp.normal('x', 4.9, 0.5), algo=tpe.suggest,
            max_evals = 2000)

print(best)
