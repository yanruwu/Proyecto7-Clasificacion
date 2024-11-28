import math
import seaborn as sns
import matplotlib.pyplot as plt


def plot_rv_dists(dataframe, response_var, num_cols = [], cat_cols = []):

    number_of_cols = len(num_cols)+len(cat_cols)

    nrows = math.ceil((number_of_cols)/2)

    fig, axes = plt.subplots(nrows=nrows, ncols = 2, figsize = (20,25))

    axes = axes.flat

    for i, col in enumerate(cat_cols):
        sns.countplot(data = dataframe, y = col, ax = axes[i], hue = response_var)
        axes[i].set_xlabel("")
        axes[i].set_title(col)
    if len(cat_cols) == 0:
        i = -1
        
    for j, col in enumerate(num_cols, start = i+1):
        sns.histplot(data = dataframe, x = col, ax = axes[j], hue = response_var)
        axes[j].set_xlabel("")
        axes[j].set_title(col)
    if number_of_cols%2 != 0 :
        plt.delaxes(axes[-1])

    plt.tight_layout()
    plt.show()
