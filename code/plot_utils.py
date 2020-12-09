import matplotlib.pyplot as plt

SMALL_SIZE = 50
MEDIUM_SIZE = 60
BIGGER_SIZE = 70


def get_epoch_avg(l, n):
    return [sum(l[i:i + n]) / n for i in range(0, len(l), n)]


def plot_history(train_history, test_history):
    fig, ax = plt.subplots()
    fig.set_size_inches(80, 15)
    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    plt.plot(train_history, color='r', lw=5, label='train')
    plt.plot(test_history, color='b', lw=5, label='test')

    plt.xlabel('epochs')

    # Don't allow the axis to be on top of your data
    ax.set_axisbelow(True)
    ax.minorticks_on()
    ax.grid(which='major', linestyle='-', linewidth='2.0', color='gray')
    ax.grid(which='minor', linestyle=':', linewidth='1.5', color='gray')
    ax.legend(bbox_to_anchor=(1.1, 1.05))
    plt.show()
