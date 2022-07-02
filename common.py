import numpy as np
import matplotlib.pyplot as plt


def array_division(a, b):
    """This function devides to arrays and enforces 0/0=0"""
    ratio = []
    for i in range(len(a)):
        if a[i] == 0:
            ratio.append(0)
        else:
            ratio.append(a[i]/b[i])
    return np.array(ratio)


def print_progressbar(value, maximum):
    """This function prints a progress bar"""
    steps = 20
    fraction = value/maximum
    progressbar = '['
    progressbar += round(fraction*steps)*'='
    progressbar += '>'
    progressbar += round((1 - fraction)*steps)*' '
    progressbar += ']'
    progressbar += f' {round(fraction*100)} percentage complete'
    print(progressbar, end='\r')


def plot_hist_sum(variable, input_data_frames):
    """This function plots the sum off the data frames for a given variable"""
    fig = plt.figure(figsize=(7, 7))
    
    # Order to plot
    process_order = ['llll', 'Zee', 'Zmumu', 'ttbar_lep', 'VBFH125_ZZ4lep', 'WH125_ZZ4lep', 'ZH125_ZZ4lep', 'ggH125_ZZ4lep']
    # Colors of processes
    process_color = {
        'llll': 'blue',
        'Zee': 'blueviolet',
        'Zmumu': 'purple',
        'ttbar_lep': 'green',
        'VBFH125_ZZ4lep': 'gold',
        'WH125_ZZ4lep': 'orange',
        'ZH125_ZZ4lep': 'sienna',
        'ggH125_ZZ4lep': 'red'
    }
    
    # Extract the histogram values 
    labels = []
    events = []
    weights = []
    colors = []
    for process in process_order:
        if process not in input_data_frames:
            continue
        values = input_data_frames[process]
        labels.append(process)
        events.append(np.array(values[variable['variable']]))
        weights.append(values['totalWeight'])
        colors.append(process_color[process])
    
    # Create the histogram
    if 'binning' in variable:
        hist_signal = plt.hist(events,
                               weights=weights,
                               bins=variable['binning'],
                               label=labels,
                               color=colors,
                               stacked=True)
            
    else:
        hist_signal = plt.hist(events,
                               weights=weights,
                               label=labels,
                               color=colors,
                               stacked=True)
    # Style
    plt.xlim(hist_signal[1][0], hist_signal[1][-1])
    plt.ylim(bottom=0)
    if 'xlabel' in variable:
        plt.xlabel(variable['xlabel'])
    else:
        plt.xlabel(variable['variable'])
    plt.ylabel('Events')
    plt.savefig('plots/hist_{}.pdf'.format(variable['variable']))
    plt.legend()
    plt.show()


def plot_normed_signal_vs_background(variable, data_frame_signal, data_frame_background):
    """This function plots the normalized signal and background distribution for a given variable"""
    total_signal = sum(data_frame_signal.totalWeight)
    total_background = sum(data_frame_background.totalWeight)

    # Set up subplots
    fig = plt.figure(figsize=(7, 7))
    gs = fig.add_gridspec(2, hspace=0.1, height_ratios=[4, 1])
    axes = gs.subplots(sharex=True, sharey=False)

    # Main plot
    if 'binning' in variable:
        binning = variable['binning']
        hist_signal = axes[0].hist(data_frame_signal[variable['variable']],
                                   weights=data_frame_signal.totalWeight / total_signal,
                                   bins=binning,
                                   label='signal',
                                   histtype='step')
        hist_background = axes[0].hist(data_frame_background[variable['variable']],
                                       weights=data_frame_background.totalWeight / total_background,
                                       bins=binning,
                                       label='background',
                                       histtype='step')
    else:
        hist_signal = axes[0].hist(data_frame_signal[variable['variable']],
                                   weights=data_frame_signal.totalWeight / total_signal,
                                   label='signal',
                                   histtype='step')
        hist_background = axes[0].hist(data_frame_background[variable['variable']],
                                       weights=data_frame_background.totalWeight / total_background,
                                       label='background',
                                       histtype='step')
        binning = hist_signal[1]

    bin_center = [(binning[i] + binning[i + 1]) / 2. for i in range(len(binning) - 1)]
    # Ratio plot
    axes[1].hist(bin_center,
                 weights=array_division(hist_signal[0], (hist_signal[0] + hist_background[0])),
                 bins=binning,
                 histtype='step')
    
    # Style
    plt.xlim([binning[0], binning[-1]])
    axes[0].set_title('Signal and background {} distribution'.format(variable['variable']))
    axes[0].set(ylabel='event fraction')
    axes[0].legend()
    if 'xlabel' in variable:
        axes[1].set(xlabel=variable['xlabel'], ylabel='normed S/(S+B)')
    else:
        axes[1].set(xlabel=variable['variable'], ylabel='normed S/(S+B)')
    axes[1].set_ylim([0, 1.4])
    axes[1].axhline(y=0.5, color='k', linestyle='-', linewidth=0.7)

    # Non-overlap
    non_overlap = sum(abs(hist_signal[0] - hist_background[0]))/2.
    axes[1].text(0.01, 0.95, 'Non-overlap = {}'.format(np.round(non_overlap, 4)),
                 verticalalignment='top', transform=axes[1].transAxes)

    plt.savefig('plots/normed_signal_vs_background_{}.pdf'.format(variable['variable']))
    plt.show()
