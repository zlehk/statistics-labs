import scipy.stats as ss
import numpy as np
import matplotlib.pyplot as plt

n = 20


def lsc(smp_x, smp_y):
    """Least Square Criteria"""
    e_x, e_y = sum(smp_x) / n, sum(smp_y) / n
    smp_x2 = [i ** 2 for i in smp_x]
    e_xy, e_x2 = sum(np.multiply(smp_x, smp_y)) / n, sum(smp_x2) / n
    b = (e_xy - e_x * e_y) / (e_x2 - e_x ** 2)
    a = e_y - e_x * b
    return {'a': a, 'b': b}


def lad(smp_x, smp_y):
    """Least Absolute Deviations"""

    def med(smp):
        return (smp[n // 2 - 1] + smp[n // 2]) / 2 if n % 2 == 0 else smp[(n - 1) // 2]

    def sgn(num):
        return 1 if num > 0 else (0 if num == 0 else -1)

    med_x, med_y = med(smp_x), med(smp_y)
    sgn_x = [sgn(i - med_x) for i in smp_x]
    sgn_y = [sgn(i - med_y) for i in smp_y]
    r_Q = sum(np.multiply(sgn_x, sgn_y)) / n
    _l = n // 4 if n % 4 == 0 else n // 4 + 1
    j = n - _l + 1
    qy_by_qx = (smp_y[j] - smp_y[_l]) / (smp_x[j] - smp_x[_l])
    b = r_Q * qy_by_qx
    a = med_y - b * med_x
    return {'a': a, 'b': b}


e = ss.norm.rvs(size=n, loc=0, scale=1)
x = np.arange(-1.8, 2 + 1e-10, 0.2)
y1 = 2 + 2 * x + e
y2 = 2 + 2 * x + e
y2[0] += 10
y2[n - 1] -= 10


def draw_results(y, name):
    fig, ax = plt.subplots(1, 1)

    plt.plot(x, 2 + 2 * x, color='#000000')
    plt.plot(x, y, color='#AA4D71', marker='o', linestyle='')
    lsc_res, lad_res = lsc(x, y), lad(x, y)
    plt.plot(x, lsc_res['a'] + lsc_res['b'] * x, color='#1EAA8C')
    plt.plot(x, lad_res['a'] + lad_res['b'] * x, color='#AA7C0F')

    ax.set_title(name)
    ax.set_ylabel('y')
    ax.set_xlabel('x')
    ax.set_facecolor('#d8dcd6')
    ax.legend(['Модель', 'Выборка', 'МНК', 'МНМ'])
    plt.show()
    print(name)
    print('МНК: a={:.4f}, b={:.4f}'.format(lsc_res['a'], lsc_res['b']))
    print('МНМ: a={:.4f}, b={:.4f}'.format(lad_res['a'], lad_res['b']))


draw_results(y1, 'Выборка без возмущений')
draw_results(y2, 'Выборка с возмущениями')
