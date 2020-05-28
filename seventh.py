import scipy.stats as ss
import numpy as np
import pandas as pd
import math


def get_normal_MLE_estimates(smp):
    e = sum(smp) / smp.size
    d = sum([(x - e) ** 2 for x in smp]) / smp.size
    return [e, d]


def get_gaps(k):
    ticks = k - 1
    center_point, gap_size = np.random.rand(), np.random.rand()
    gap_size = gap_size + 0.5 if gap_size < 0.5 else gap_size
    right_side = (ticks - 1) // 2
    left_side = ticks - 1 - right_side
    gaps = [center_point]
    for i in range(int(right_side)):
        gaps.append(center_point + (i + 1) * gap_size)
    for i in range(int(left_side)):
        gaps.append(center_point - (i + 1) * gap_size)
    gaps.sort()
    res_gaps = [(-math.inf, gaps[0])]
    for i in range(int(ticks - 1)):
        res_gaps.append((gaps[i], gaps[i + 1]))
    res_gaps.append((gaps[int(ticks - 1)], math.inf))
    return res_gaps


def chi_squared_hypothesis_test(smp, alpha=0.05, table='without'):
    def get_Sturges_k():
        return 1 + 3.3 * np.log10(smp.size)

    def get_hist_k():
        return 1.72 * (smp.size ** (1 / 3))

    k = np.round((get_Sturges_k() + get_hist_k()) / 2)
    df = pd.DataFrame(columns=['intrvl_bgn', 'intrvl_end'], data=get_gaps(k))
    df['ni'] = df.apply(lambda row: np.sum(smp <= row.intrvl_end) - np.sum(smp <= row.intrvl_bgn), axis=1)
    df['pi'] = df.apply(lambda row: ss.norm.cdf(row.intrvl_end) - ss.norm.cdf(row.intrvl_bgn), axis=1)
    df['npi'] = df.apply(lambda row: row.pi * smp.size, axis=1)
    df['ni_npi'] = df.apply(lambda row: row.ni - row.npi, axis=1)
    df['frac'] = df.apply(lambda row: row.ni_npi ** 2 / row.npi, axis=1)
    df = df.append(df.sum().rename('total'))
    df['ni_npi']['total'] = 0
    if table == 'with':
        print(df)
    return {'estimate': df['frac']['total'], 'alpha_quant': ss.chi2.ppf(1 - alpha, k - 1)}


def research(reps=1000):
    print('Research start:---')
    res_smp = []
    res_bool = np.array([])
    for _ in range(reps):
        smp = ss.norm.rvs(size=n, loc=0, scale=1)
        res = chi_squared_hypothesis_test(smp)
        res_smp.append(res['estimate'])
        res_bool = np.append(res_bool, [res['estimate'] < res['alpha_quant']])

    e = sum(res_smp) / reps
    res_smp = [x ** 2 for x in res_smp]
    e2 = sum(res_smp) / reps
    d = e2 - e ** 2

    print('e={:.4f}, d={:.4f}'.format(e, d))
    print('True/Total= {}'.format(np.sum(np.where(res_bool, 1, 0)) / res_bool.size))


n = 100
sample = ss.norm.rvs(size=n, loc=0, scale=1)
print(get_normal_MLE_estimates(sample))
chi2_result = chi_squared_hypothesis_test(sample, table='with')
print('{:.4f} < {:.4f}: {}'.format(chi2_result['estimate'], chi2_result['alpha_quant'],
                                   chi2_result['estimate'] < chi2_result['alpha_quant']))

research()
