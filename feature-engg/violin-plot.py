import numpy as np
import matplotlib.pyplot as plt

n_datasets = 10
n_samples = 730
data = np.random.randn(n_datasets,n_samples)

fig, axes = plt.subplots(1,3)

# http://matplotlib.org/examples/statistics/boxplot_vs_violin_demo.html
axes[0].violinplot([d for d in data])

# should be equivalent to:
axes[1].violinplot(data)

# is actually equivalent to
