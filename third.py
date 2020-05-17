import matplotlib.pyplot as plt
import pandas as pd
from distribs import distrib_sample, distrib_names


def draw_boxplot(name):
    plt.figure()
    fig, ax = plt.subplots(1, 1)

    df20 = pd.DataFrame({'20': distrib_sample(name, 20)})
    df100 = pd.DataFrame({'100': distrib_sample(name, 100)})
    df = pd.concat([df20, df100], axis=1)
    df.boxplot(vert=False, widths=(0.6, 0.6), color={'medians': '#AA4D71'}, medianprops={'linewidth': 2})

    ax.set_title(name)
    ax.set_ylabel('n')
    ax.set_xlabel('x')

    plt.show(block=False)


def get_outlier_fraction(name, n=1000):
    outliers20, outliers100 = [], []
    for _ in range(n):
        smp = {'20': distrib_sample(name, 20), '100': distrib_sample(name, 100)}
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
    e20, e100 = sum(outliers20) / n, sum(outliers100) / n
    outliers20 = [x ** 2 for x in outliers20]
    outliers100 = [x ** 2 for x in outliers100]
    d20, d100 = (sum(outliers20) / n) - e20 ** 2, (sum(outliers100) / n) - e100 ** 2
    print("20: e={:.4f} d={:.4f}".format(e20, d20))
    print("100: e={:.4f} d={:.4f}".format(e100, d100))


for i in range(5):
    draw_boxplot(distrib_names[i])
    get_outlier_fraction(distrib_names[i])
