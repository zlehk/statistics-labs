import scipy.stats as ss
import numpy as np

distribs = {'normal': {'d': ss.norm, 'scale': 1},
            'cauchy': {'d': ss.cauchy, 'scale': 1},
            'laplace': {'d': ss.laplace, 'scale': 1 / (2 ** 0.5)},
            'poisson': {'d': ss.poisson, 'scale': 10},
            'uniform': {'d': ss.uniform, 'scale': 3 ** 0.5}}
distrib_names = ['Normal', 'Cauchy', 'Laplace', 'Poisson', 'Uniform']


def distrib_sample(name, n):
    if name.lower() == 'normal':
        return distribs['normal']['d'].rvs(size=n)
    elif name.lower() == 'cauchy':
        return distribs['cauchy']['d'].rvs(size=n)
    elif name.lower() == 'laplace':
        return distribs['laplace']['d'].rvs(size=n, scale=distribs['laplace']['scale'])
    elif name.lower() == 'poisson':
        return distribs['poisson']['d'].rvs(size=n, mu=distribs['poisson']['scale'])
    elif name.lower() == 'uniform':
        return distribs['uniform']['d'].rvs(size=n, loc=-distribs['uniform']['scale'],
                                            scale=distribs['uniform']['scale'] * 2)
    else:
        return np.array([0])
