import numpy as np
import scipy.stats as ss
from tabulate import tabulate
import math


def metrics(rv, n):
    rv.sort()

    def mean():
        return sum(rv) / n

    def med():
        return (rv[n // 2 - 1] + rv[n // 2]) / 2 if n % 2 == 0 else rv[(n - 1) // 2]

    def z_r():
        return (rv[0] + rv[n - 1]) / 2

    def z_q():
        def z_p(p):
            return rv[int(n * p) - 1] if (n * p) % 1 == 0 else rv[math.floor(n * p)]

        return (z_p(1 / 4) + z_p(3 / 4)) / 2

    def z_tr():
        r = n // 4
        return sum(rv[r:n - r - 1]) / (n - 2 * r)

    return np.array([mean(), med(), z_r(), z_q(), z_tr()])


sample_sizes = [10, 100, 1000]
reps = 1000
columns = ['n', 'mean', 'median', 'z_r', 'z_q', 'z_tr']
names = ['Normal', 'Cauchy', 'Laplace', 'Poisson', 'Uniform']
exp_values, variances = [], []

for i in range(len(sample_sizes)):
    sample = np.array([[0, 0, 0, 0, 0]])
    for _ in range(reps):
        '''Пять возможных вариантов - это пять распределений с параметрами'''
        sample = np.append(sample, [metrics(ss.norm.rvs(size=sample_sizes[i]), sample_sizes[i])], axis=0)
        # sample = np.append(sample, [metrics(ss.cauchy.rvs(size=sample_sizes[i]), sample_sizes[i])], axis=0)
        # sample = np.append(sample,
        #                    [metrics(ss.laplace.rvs(scale=(1 / (2 ** 0.5)), size=sample_sizes[i]), sample_sizes[i])],
        #                    axis=0)
        # sample = np.append(sample, [metrics(ss.poisson.rvs(mu=10, size=sample_sizes[i]), sample_sizes[i])], axis=0)
        # sample = np.append(sample,
        #                    [metrics(ss.uniform.rvs(loc=-(3 ** .5), scale=2 * (3 ** .5), size=ns[i]), ns[i])], axis=0)
    es = [sample_sizes[i]]
    ds = [sample_sizes[i]]
    for j in range(len(columns) - 1):
        met = np.take(sample, j, axis=1)
        e = sum(met) / reps
        es.append(e)

        met = [x ** 2 for x in met]
        d = (sum(met) / reps) - e ** 2
        ds.append(d)
    exp_values.append(es)
    variances.append(ds)

print("E: ")
print(tabulate(exp_values, headers=columns, floatfmt=".6f", tablefmt='github'))
print("D: ")
print(tabulate(variances, headers=columns, floatfmt=".6f", tablefmt='github'))

