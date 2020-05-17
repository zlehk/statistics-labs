import scipy.stats as ss
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def dist_sample(name, n):
    if name.lower() == 'normal':
        return ss.norm.rvs(size=n)
    elif name.lower() == 'cauchy':
        return ss.cauchy.rvs(size=n)
    elif name.lower() == 'laplace':
        return ss.laplace.rvs(size=n, scale=scales['laplace'])
    elif name.lower() == 'poisson':
        return ss.poisson.rvs(size=n, mu=scales['poisson'])
    elif name.lower() == 'uniform':
        return ss.uniform.rvs(size=n, loc=-scales['uniform'], scale=scales['uniform'] * 2)
    else:
        return np.array([])


def draw_boxplot(name):
    plt.figure()
    fig, ax = plt.subplots(1, 1)

    df20 = pd.DataFrame({'20': dist_sample(name, 20)})
    df100 = pd.DataFrame({'100': dist_sample(name, 100)})
    df = pd.concat([df20, df100], axis=1)
    df.boxplot(vert=False, widths=(0.6, 0.6), color={'medians': '#AA4D71'}, medianprops={'linewidth': 2})

    ax.set_title(name)
    ax.set_ylabel('n')
    ax.set_xlabel('x')

    plt.show(block=False)


def get_outlier_fraction(name, n=1000):
    outliers20, outliers100 = [], []
    for _ in range(n):
        smp = {'20': dist_sample(name, 20), '100': dist_sample(name, 100)}
        df20 = pd.DataFrame({'20': smp['20']})
        df100 = pd.DataFrame({'100': smp['100']})
        df = pd.concat([df20, df100], axis=1)
        q1, q3 = df.quantile(q=.25), df.quantile(q=.75)
        x1, x2 = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)

        outliers20.append(
            (len(smp['20'][smp['20'] < x1['20']]) + len(smp['20'][smp['20'] > x2['20']]))
            / len(smp['20'])
        )
        outliers100.append(
            (len(smp['100'][smp['100'] < x1['100']]) + len(smp['100'][smp['100'] > x2['100']]))
            / len(smp['100'])
        )
    print(name)
    print("{:.4f}".format(sum(outliers20) / n))
    print("{:.4f}".format(sum(outliers100) / n))


dists = [ss.norm, ss.cauchy, ss.laplace, ss.poisson, ss.uniform]
names = ['Normal', 'Cauchy', 'Laplace', 'Poisson', 'Uniform']
scales = {'normal': 1, 'cauchy': 1, 'laplace': 1 / (2 ** 0.5), 'poisson': 10, 'uniform': 3 ** 0.5}

for i in range(len(dists)):
    draw_boxplot(names[i])
    get_outlier_fraction(names[i])
