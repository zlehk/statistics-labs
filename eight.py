import scipy.stats as ss
import pandas as pd

alpha = 0.05
sample20 = ss.norm.rvs(size=20)
sample100 = ss.norm.rvs(size=100)


def m_MLE(smp):
    df = pd.DataFrame(data=smp, columns=['sample'])
    df = df.append(df.mean().rename('mean'))
    df = df.append(df.std().rename('std'))
    frac = (df['sample']['std'] * ss.t.ppf(1 - alpha / 2, smp.size - 1)) / ((smp.size - 1) ** 0.5)
    return [df['sample']['mean'] - frac, df['sample']['mean'] + frac]


def sigma_MLE(smp):
    df = pd.DataFrame(data=smp, columns=['sample'])
    df = df.append(df.std().rename('std'))
    numerator = df['sample']['std'] * (smp.size ** 0.5)
    return [-numerator / (ss.chi2.ppf(1 - alpha / 2, smp.size - 1) ** 0.5),
            numerator / (ss.chi2.ppf(alpha / 2, smp.size - 1) ** 0.5)]


m20, sigma20 = m_MLE(sample20), sigma_MLE(sample20)
print('n=20: MLE {:.4f} < m < {:.4f} & {:.4f} < sigma < {:.4f}'.format(m20[0], m20[1], sigma20[0], sigma20[1]))
m100, sigma100 = m_MLE(sample100), sigma_MLE(sample100)
print('n=100: MLE {:.4f} < m < {:.4f} & {:.4f} < sigma < {:.4f}'.format(m100[0], m100[1], sigma100[0], sigma100[1]))


def m_Asymptotic(smp):
    df = pd.DataFrame(data=smp, columns=['sample'])
    df = df.append(df.mean().rename('mean'))
    df = df.append(df.std().rename('std'))
    frac = (df['sample']['std'] * ss.norm.ppf(1 - alpha / 2)) / (smp.size ** 0.5)
    return [df['sample']['mean'] - frac, df['sample']['mean'] + frac]


def sigma_Asymptotic(smp):
    df = pd.DataFrame(data=smp, columns=['sample'])
    _x = df['sample'].mean()
    df['m4'] = df.apply(lambda row: (row - _x) ** 4, axis=1)
    df = df.append(df.mean().rename('mean'))
    df = df.append(df.std().rename('std'))
    # print(df)
    e = df['m4']['mean'] / (df['sample']['std'] ** 4) - 3
    U = ss.norm.ppf(1 - alpha / 2) * (((e + 2) / smp.size) ** 0.5)
    return [df['sample']['std'] / ((1 + U) ** 0.5), df['sample']['std'] / ((1 - U) ** 0.5)]


m20, sigma20 = m_Asymptotic(sample20), sigma_Asymptotic(sample20)
print('n=20: Asymp {:.4f} < m < {:.4f} & {:.4f} < sigma < {:.4f}'.format(m20[0], m20[1], sigma20[0], sigma20[1]))
m100, sigma100 = m_Asymptotic(sample100), sigma_Asymptotic(sample100)
print('n=100: Asymp {:.4f} < m < {:.4f} & {:.4f} < sigma < {:.4f}'.format(m100[0], m100[1], sigma100[0], sigma100[1]))
