import scipy.stats as ss
import numpy as np
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


def get_r_estimates(smp):
    df = pd.DataFrame(data=smp, columns=['X', 'Y'])
    n = np.size(smp, axis=0)

    def pearson():
        K = df.cov()
        return K['X']['Y'] / (df['X'].std() * df['Y'].std())

    def quadrant():
        medX, medY = df['X'].median(), df['Y'].median()
        n1 = len([p for p in df.values if p[0] > medX and p[1] >= medY])
        n2 = len([p for p in df.values if p[0] <= medX and p[1] > medY])
        n3 = len([p for p in df.values if p[0] < medX and p[1] <= medY])
        n4 = len([p for p in df.values if p[0] >= medX and p[1] < medY])
        return (n1 + n3 - n2 - n4) / n

    def spearmen():
        sp_df = df.sort_values(by='X')
        sp_df['u'] = range(1, n + 1)
        sp_df = sp_df.sort_values(by='Y')
        sp_df['v'] = range(1, n + 1)
        sp_df = sp_df.sort_index()
        K = sp_df[['u', 'v']].cov()
        return K['u']['v'] / (sp_df['u'].std() * sp_df['v'].std())

    return [pearson(), spearmen(), quadrant()]


def get_e_e2_d(smp):
    np_smp = np.array(smp)
    table = np.empty((3, 0))
    for i in range(np.size(np_smp, axis=1)):
        col = np.take(np_smp, i, axis=1)
        e = sum(col) / reps

        col = [x ** 2 for x in col]
        e2 = sum(col) / reps

        d = e2 - e ** 2
        table = np.concatenate([table, np.array([[e], [e2], [d]])], axis=1)
    return table


reps = 1000
columns = ['r', 'rS', 'rQ']


def normal_distribution_with_rho():
    for n in ns:
        for c in rhos:
            rs = []
            for _ in range(reps):
                sample = ss.multivariate_normal.rvs(mean=np.array([0, 0]), cov=np.array([[1, c], [c, 1]]), size=n)
                rs.append(get_r_estimates(sample))
            ee2d = get_e_e2_d(rs)
            print("n={} rho={}".format(n, c))
            print(tabulate(ee2d, headers=columns, floatfmt=".6f", tablefmt='github'))


def mixed_distribution():
    print("Mixed distribution")
    for n in ns:
        rs = []
        for _ in range(reps):
            sample = 0.9 * ss.multivariate_normal.rvs(mean=np.array([0, 0]), cov=np.array([[1, 0.9], [0.9, 1]]),
                                                      size=n) + \
                     0.1 * ss.multivariate_normal.rvs(mean=np.array([0, 0]), cov=np.array([[10, -9], [-9, 10]]), size=n)
            rs.append(get_r_estimates(sample))
        ee2d = get_e_e2_d(rs)
        print("n={}".format(n))
        print(tabulate(ee2d, headers=columns, floatfmt=".6f", tablefmt='github'))


def draw_dispersion_area(size, rho):
    fig, ax = plt.subplots(1, 1)
    plt.axis([-3, 3, -3, 3])

    sample = ss.multivariate_normal.rvs(mean=np.array([0, 0]), cov=np.array([[1, rho], [rho, 1]]), size=size)

    df = pd.DataFrame(data=sample, columns=['X', 'Y'])
    p_rho = rho
    sX = df['X'].std()
    sY = df['Y'].std()
    alpha = np.arctan((2 * p_rho * sX * sY) / (sX ** 2 - sY ** 2)) / 2
    sE = (sX * np.cos(alpha)) ** 2 + p_rho * sX * sY * np.sin(2 * alpha) + (sY * np.sin(alpha)) ** 2
    sN = (sX * np.sin(alpha)) ** 2 - p_rho * sX * sY * np.sin(2 * alpha) + (sY * np.cos(alpha)) ** 2
    k = 2.45

    ellipse = Ellipse((0, 0),
                      width=np.sqrt(sE) * k * 2,
                      height=np.sqrt(sN) * k * 2,
                      fill=False,
                      edgecolor='#025052')

    transf = transforms.Affine2D() \
        .rotate_deg(alpha * (180 / np.pi))

    ellipse.set_transform(transf + ax.transData)
    ax.add_patch(ellipse)

    plt.plot(np.take(sample, 0, axis=1), np.take(sample, 1, axis=1), color='#AA4D71', marker='o',
             linestyle='')

    ax.set_title('n={}, rho={}'.format(size, rho))
    ax.set_ylabel('y')
    ax.set_xlabel('x')
    ax.set_facecolor('#d8dcd6')
    plt.show()


rhos = [0, 0.5, 0.9]
ns = [20, 60, 100]

for _n in ns:
    for _rho in rhos:
        draw_dispersion_area(_n, _rho)
