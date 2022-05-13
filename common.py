import numpy as np
import matplotlib.pyplot as plt

def array_division(a, b):
    ratio = []
    for i in range(len(a)):
        if b[i] == 0:
            ratio.append(0)
        else:
            ratio.append(a[i]/b[i])
    return np.array(ratio)

def plot_hist(variable, data_frame_signal, data_frame_background):
    total_signal = sum(data_frame_signal.weight)
    total_background = sum(data_frame_background.weight)
    bin_center = [(variable['bins'][i] + variable['bins'][i + 1]) / 2. for i in range(len(variable['bins']) - 1)]

    # Set up subplots
    fig = plt.figure(figsize=(7, 7))
    gs = fig.add_gridspec(2, hspace=0.1, height_ratios=[4, 1])
    axes = gs.subplots(sharex=True, sharey=False)

    # Main plot
    hist_signal = axes[0].hist(data_frame_signal[variable['variable']],
                               weights=data_frame_signal.weight / total_signal,
                               bins=variable['bins'],
                               label='signal',
                               histtype='step')
    hist_background = axes[0].hist(data_frame_background[variable['variable']],
                                   weights=data_frame_background.weight / total_background,
                                   bins=variable['bins'],
                                   label='background',
                                   histtype='step')

    # Ratio plot
    axes[1].hist(bin_center,
                 weights=array_division(hist_signal[0], (hist_signal[0] + hist_background[0])),
                 bins=variable['bins'],
                 histtype='step')
    
    # Style
    plt.xlim([variable['bins'][0], variable['bins'][-1]])
    axes[0].set_title('Signal and background {} distribution'.format(variable['variable']))
    axes[0].set(ylabel='event fraction')
    axes[0].legend()
    axes[1].set(xlabel=variable['xlabel'], ylabel='normed S/(S+B)')
    axes[1].set_ylim([0, 1.4])
    axes[1].axhline(y=0.5, color='k', linestyle='-', linewidth=0.7)

    # Non-overlap
    non_overlap = sum(abs(hist_signal[0] - hist_background[0]))/2.
    axes[1].text(0.01, 0.95, 'Non-overlap = {}'.format(np.round(non_overlap, 4)),
                 verticalalignment='top', transform=axes[1].transAxes)

    plt.savefig('plots/investigate_{}.pdf'.format(variable['variable']))
    plt.show()
