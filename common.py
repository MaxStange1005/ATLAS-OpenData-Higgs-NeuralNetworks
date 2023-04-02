import numpy as np
import matplotlib.pyplot as plt
import itertools


def get_bin_center(bins):
    """Get the center values of the given array"""
    center = (bins[1:] + bins[:-1]) / 2
    return center


def get_hists(bins, prediction, classification, weight=None, norm=True):
    """Create a histogram of the signal and background prediction"""
    prediction = np.array(list(itertools.chain(*prediction)))
    # Split by prediction
    prediction_signal = prediction[classification == 1]
    prediction_bkg = prediction[classification == 0]
    if not weight:
        # Create Histogram
        hist_signal = np.histogram(prediction_signal, bins)[0]
        hist_bkg = np.histogram(prediction_bkg, bins)[0]
    else:
        weight_signal = weight[classification == 1]
        weight_bkg = weight[classification == 0]
        # Create Histogram
        hist_signal = np.histogram(prediction_signal, bins, weights=weight_signal)[0]
        hist_bkg = np.histogram(prediction_bkg, bins, weights=weight_bkg)[0]

    # Normalize histogram
    if norm:
        hist_signal = hist_signal / hist_signal.sum()
        hist_bkg = hist_bkg / hist_bkg.sum()
    return hist_signal, hist_bkg


def plot_dnn_output(train_prediction, train_classification, val_prediction=None, val_classification=None):
    """Create a figure with the DNN classification on train and validation data"""
    # Create bins
    bins = np.linspace(0, 1, 41)
    bins_center = get_bin_center(bins)

    fig, ax = plt.subplots(figsize=(7, 6))
    # Histograms for train data
    hist_train_signal, hist_train_bkg = get_hists(bins, train_prediction, train_classification)
    ax.hist(bins_center, bins=bins, weights=hist_train_signal,
                 histtype='step', label='train signal', color='b')
    ax.hist(bins_center, bins=bins, weights=hist_train_bkg,
                 histtype='step', label='train bkg', color='darkorange')
    # Histograms for val data
    if val_prediction is not None and val_classification is not None:
        hist_val_signal, hist_val_bkg = get_hists(bins, val_prediction, val_classification)
        ax.plot(bins_center, hist_val_signal, label='val signal', marker='.', ls='', color='b')
        ax.plot(bins_center, hist_val_bkg, label='val bkg', marker='.', ls='', color='darkorange')
    ax.legend()
    return fig


def get_dnn_input(data_frames, training_variables, sample_list_signal, sample_list_background, frac=1):
    """This function extracts the training values, weights and classification of all signal and background samples.
    If you just want to use a fraction of the data use frac"""
    values = []
    weights = []
    classification = []
    for sample in sample_list_signal + sample_list_background:
        data = data_frames[sample]
        # Use just a fraction of the data:
        if frac < 1:
            data = data[np.random.rand(len(data)) < frac]
        # Classify signal and background (and skip if data)
        if sample in sample_list_signal:
            # 1 if signal
            classification.append(np.ones(len(data)))
        elif sample in sample_list_background:
            # 0 if background
            classification.append(np.zeros(len(data)))
        else:
            continue
        # input values
        values.append(data[training_variables])
        weights.append(data['totalWeight'])

    # Merge the input
    values = np.concatenate(values)
    weights = np.concatenate(weights)
    classification = np.concatenate(classification)
    return values, weights, classification


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


def plot_hist(variable, input_data_frames, show_data=True):
    """This function plots the sum off the data frames for a given variable"""
    fig = plt.figure(figsize=(7, 7))
    if show_data:
        gs = fig.add_gridspec(2, hspace=0.1, height_ratios=[4, 1])
        axes = gs.subplots(sharex=True, sharey=False)
    else:
        axes = [plt.axes()]
    
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
        hist_MC = axes[0].hist(events,
                               weights=weights,
                               bins=variable['binning'],
                               label=labels,
                               color=colors,
                               stacked=True)
    else:
        hist_MC = axes[0].hist(events,
                               weights=weights,
                               label=labels,
                               color=colors,
                               stacked=True)

    if show_data:
        # Measured data
        measured_data = []
        for sample in ['data_A', 'data_B', 'data_C', 'data_D']:
            measured_data += list(input_data_frames[sample][variable['variable']])
        if 'binning' in variable:
            data_num, bin_edges = np.histogram(measured_data, bins=variable['binning'])
        else:
            data_num, bin_edges = np.histogram(measured_data)
        bin_center = 0.5*(bin_edges[1:] + bin_edges[:-1])
        axes[0].errorbar(
            bin_center,
            data_num,
            yerr = np.sqrt(data_num),
            color='k',
            label='data',
            ls='none',
            marker = '.',
        )
        # Ratio plot
        axes[1].errorbar(
            bin_center,
            array_division(data_num, hist_MC[0][-1]),
            yerr = array_division(np.sqrt(data_num), hist_MC[0][-1]),
            color='k',
            ls='none',
            marker = '.',
        )
        axes[1].axhline(1)
        axes[1].set(ylabel='data/MC')
        axes[1].set_ylim([0, 2])
    
    # Style
    if 'binning' in variable:
        plt.xlim(variable['binning'][0], variable['binning'][-1])
    else:
        plt.xlim(hist_signal[1][0], hist_signal[1][-1])
    axes[0].set_title('{} distribution'.format(variable['variable']))
    axes[0].set(ylabel='Events')
    axes[0].set_ylim(bottom=0)
    axes[0].legend()
    if 'xlabel' in variable:
        axes[-1].set(xlabel=variable['xlabel'])
    else:
        axes[-1].set(xlabel=variable['variable'])
    #plt.savefig('plots/hist_{}.pdf'.format(variable['variable']))
    plt.show()
    return 


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

    #plt.savefig('plots/normed_signal_vs_background_{}.pdf'.format(variable['variable']))
    plt.show()
