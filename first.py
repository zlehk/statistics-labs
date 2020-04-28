import scipy.stats as ss
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def draw_hist(dist, name, scale, n):
    fig, ax = plt.subplots(1, 1)

    if dist is ss.laplace:
        sample = dist.rvs(size=n, scale=scale)
    elif dist is ss.poisson:
        sample = dist.rvs(size=n, mu=scale)
    elif dist is ss.uniform:
        sample = dist.rvs(size=n, loc=-scale, scale=scale * 2)
    else:
        sample = dist.rvs(size=n)

    data = pd.DataFrame(sample)
    data.plot.hist(ax=ax, density=True, legend=False, title=name + 'Numbers n={}'.format(n), color='#AA6C7F',
                   edgecolor='black', linewidth=1)

    if dist is ss.poisson:
        x = np.arange(min(sample) - 3, max(sample) + 3, 1)
        dens = pd.DataFrame(dist.pmf(x, mu=scale), x)
    else:
        x = np.linspace(min(sample), max(sample), 200)
        if dist is ss.uniform:
            dens = pd.DataFrame(dist.pdf(x, loc=-scale, scale=scale * 2), x)
        elif dist is ss.laplace:
            dens = pd.DataFrame(dist.pdf(x, scale=scale), x)
        else:
            dens = pd.DataFrame(dist.pdf(x), x)
    dens.plot.line(ax=ax, legend=False, color='#13AA83', linewidth=2)

    ax.set_ylabel('Density')
    ax.set_xlabel(name + 'Numbers')
    ax.grid(axis='y')
    ax.set_facecolor('#d8dcd6')
    plt.show()


dists = [ss.norm, ss.cauchy, ss.laplace, ss.poisson, ss.uniform]
names = ['Normal', 'Cauchy', 'Laplace', 'Poisson', 'Uniform']
scales = [1, 1, 1 / (2 ** 0.5), 10, 3 ** 0.5]
ns = [10, 50, 1000]
for i in range(len(dists)):
    for j in range(len(ns)):
        draw_hist(dists[i], names[i], scales[i], ns[j])
