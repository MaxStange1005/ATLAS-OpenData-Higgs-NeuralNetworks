import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import figure
import itertools
from typing import Union, Tuple, List, Dict
from keras import Sequential

# Random state
random_state = 21
_ = np.random.RandomState(random_state)


def merge_data_frames(sample_list: Union[list, pd.DataFrame], data_frames_dic: Dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Merge the signal and background dataframes.

    Parameters:
        sample_list (Union[list, np.ndarray]): List of samples to be merged.
        data_frames_dic (Dict[str, np.ndarray]): A dictionary with sample names and their corresponding pandas dataframes.

    Returns:
        pd.DataFrame: Merged dataframes.
    """
    for sample in sample_list:
        if sample == sample_list[0]:
            output_data_frame = data_frames_dic[sample]
        else:
            output_data_frame = pd.concat([output_data_frame, data_frames_dic[sample]], axis=0)
    return output_data_frame


def _get_bin_center(bins: np.ndarray) -> np.ndarray:
    """
    Get the center values of the given array.

    Parameters:
        bins (np.ndarray): Bin edges.

    Returns:
        np.ndarray: Centers of the given bin edges.
    """
    center = (bins[1:] + bins[:-1]) / 2
    return center


def _get_hists(bins: np.ndarray, prediction: np.ndarray, classification: np.ndarray, weight: np.ndarray = None,
               norm: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Create histograms of the signal and background prediction.

    Parameters:
        bins (np.ndarray): Bin edges.
        prediction (np.ndarray): Prediction by the model (array of arrays).
        classification (np.ndarray): Array of true classifications.
        weight (np.ndarray, optional): Event weights.
        norm (bool, optional): Normalize the histograms.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Classification histograms of signal and background events.
    """
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

    # Normalize histograms
    if norm:
        hist_signal = hist_signal / hist_signal.sum()
        hist_bkg = hist_bkg / hist_bkg.sum()
    return hist_signal, hist_bkg


def apply_dnn_model(model: Sequential, data_frames: Dict[str, pd.DataFrame], variables: List[str],
                    sample_list: List[str]) -> Dict[str, pd.DataFrame]:
    """Apply the provided Keras model to all the samples in the dataframes.

    Parameters:
        model (Sequential): The Keras model to apply.
        data_frames (Dict[str, pd.DataFrame]): A dictionary with sample names and their corresponding pandas dataframes.
        variables (List[str]): The variables used by the model.
        sample_list (List[str]): A list of sample names to apply the model on.

    Returns:
        Dict[str, pd.DataFrame]: A dictionary with sample names and their corresponding classifications
        and event weights in pandas dataframes.
    """
    data_frames_apply_dnn = {}
    for sample in sample_list:
        print(f'Applying Model for {sample}')
        # Get the values to apply the model
        values = data_frames[sample][variables]
        weights = data_frames[sample]['totalWeight']
        prediction = model.predict(values)

        # Convert prediction to array
        prediction = [element[0] for element in prediction]

        # Add the prediction for each sample
        data_frames_apply_dnn[sample] = pd.DataFrame({'model_prediction': prediction, 'totalWeight': weights})
    return data_frames_apply_dnn


def plot_dnn_output(train_prediction: np.ndarray, train_classification: np.ndarray, val_prediction: np.ndarray = None,
                    val_classification: np.ndarray = None) -> Tuple[figure.Figure, plt.Axes]:
    """Create a figure with the DNN classification on training and validation data.

    Parameters:
        train_prediction (np.ndarray): Predictions on the training data by the model (array of arrays).
        train_classification (np.ndarray): Array of true classifications of training data.
        val_prediction (np.ndarray, optional): Predictions on the validation data by the model (array of arrays).
        val_classification (np.ndarray, optional): Array of true classifications of validation data.

    Returns:
        Tuple[figure.Figure, plt.Axes]: A tuple containing the figure and axes of the created histogram.
    """
    # Create bins
    bins = np.linspace(0, 1, 41)
    bins_center = _get_bin_center(bins)

    fig, ax = plt.subplots(figsize=(7, 6))
    # Histograms for training data
    hist_train_signal, hist_train_bkg = _get_hists(bins, train_prediction, train_classification)
    ax.hist(bins_center, bins=bins, weights=hist_train_signal,
            histtype='step', label='train signal', color='b')
    ax.hist(bins_center, bins=bins, weights=hist_train_bkg,
            histtype='step', label='train background', color='darkorange')
    # Histograms for validation data
    if val_prediction is not None and val_classification is not None:
        hist_val_signal, hist_val_bkg = _get_hists(bins, val_prediction, val_classification)
        ax.plot(bins_center, hist_val_signal, label='validation signal', marker='.', ls='', color='b')
        ax.plot(bins_center, hist_val_bkg, label='validation background', marker='.', ls='', color='darkorange')
    ax.legend()
    return fig, ax


def split_data_frames(data_frames: Dict[str, pd.DataFrame],
                      frac: float) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """Split the dataframes at the given fraction and reweight the weights.

    Parameters:
        data_frames (Dict[str, pd.DataFrame]): A dictionary with sample names and their corresponding pandas dataframes.
        frac (float): The fraction used to split the dataframes.

    Returns:
        Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]: A tuple containing two dictionaries of split dataframes.
    """
    split1_df = {}
    split2_df = {}
    for name, sample in data_frames.items():
        split_criteria = np.random.rand(len(sample)) < frac
        # Split dataframes
        split1_df[name] = sample[split_criteria].copy()
        split2_df[name] = sample[~split_criteria].copy()

        # Reweight the weights
        split1_df[name]['totalWeight'] *= sample['totalWeight'].sum() / split1_df[name]['totalWeight'].sum()
        split2_df[name]['totalWeight'] *= sample['totalWeight'].sum() / split2_df[name]['totalWeight'].sum()
    return split1_df, split2_df


def get_dnn_input(data_frames: Dict[str, pd.DataFrame], variables: List[str],
                  sample_list_signal: List[str], sample_list_background: List[str],
                  frac: float = 1, invert_frac: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extracts the training values, weights, and classification of signal and background samples.

    Parameters:
        data_frames (Dict[str, pd.DataFrame]): A dictionary with sample names and their corresponding pandas dataframes.
        variables (List[str]): The list of variables to extract.
        sample_list_signal (List[str]): A list of sample names representing signals.
        sample_list_background (List[str]): A list of sample names representing backgrounds.
        frac (float, optional): The fraction of data to use (default is 1).
        invert_frac (bool, optional): If True, invert the fraction (use data outside the fraction).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing three numpy arrays:
            - The input values
            - The weights
            - The classification labels (1 for signal, 0 for background)
    """
    values = []
    weights = []
    classification = []

    for sample in sample_list_signal + sample_list_background:
        data = data_frames[sample]

        # Use just a fraction of the data
        if invert_frac:
            data = data[np.random.rand(len(data)) > frac]
        else:
            data = data[np.random.rand(len(data)) <= frac]

        # Classify signal and background (and skip if data)
        if sample in sample_list_signal:
            # 1 if signal
            classification.append(np.ones(len(data)))
        elif sample in sample_list_background:
            # 0 if background
            classification.append(np.zeros(len(data)))
        else:
            continue

        # Input values
        values.append(data[variables])
        weights.append(data['totalWeight'])

    # Merge the input
    values = np.concatenate(values)
    weights = np.concatenate(weights)
    classification = np.concatenate(classification)

    return values, weights, classification


def array_division(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """This function divides to arrays and enforces 0/0=0.

    Parameters
    ----------
    numerator : np.ndarray
        numerator of the division
    denominator : np.ndarray
        denominator of the division

    Returns
    -------
    np.ndarray
        ratio of the two given arrays
    """
    ratio = []
    for i in range(len(numerator)):
        if numerator[i] == 0:
            ratio.append(0)
        else:
            ratio.append(numerator[i] / denominator[i])
    return np.array(ratio)


def print_progressbar(value: int, maximum: int):
    """Prints a progress bar.

    Args:
        value (int): Current progress value.
        maximum (int): Maximum value for progress.

    Returns:
        None
    """
    steps = 20
    fraction = value / maximum
    progressbar = '['
    progressbar += round(fraction * steps) * '='
    progressbar += '>'
    progressbar += round((1 - fraction) * steps) * ' '
    progressbar += ']'
    progressbar += f' {round(fraction * 100)} percentage complete'
    print(progressbar, end='\r')


def plot_hist(variable: Dict[str, Union[str, List, Dict]],
              input_data_frames: Dict[str, pd.DataFrame],
              show_data: bool = False
              ) -> Tuple[figure.Figure, List[plt.Axes]]:
    """Plots the sum of dataframes for a given variable.

    Args:
        variable (dict): A dictionary containing information about the variable to be plotted.
        input_data_frames (dict): A dictionary of input dataframes.
        show_data (bool, optional): Whether to show measured data. Defaults to False.

    Returns:
        Tuple[figure.Figure, List[plt.Axes]]: A tuple containing the figure and list of axes.
    """
    fig = plt.figure(figsize=(7, 7))
    if show_data:
        gs = fig.add_gridspec(2, hspace=0.1, height_ratios=[4, 1])
        axes = gs.subplots(sharex=True, sharey=False)
    else:
        axes = [plt.axes()]

    # Order to plot
    process_order = ['llll', 'Zee', 'Zmumu', 'ttbar_lep', 'VBFH125_ZZ4lep', 'WH125_ZZ4lep', 'ZH125_ZZ4lep',
                     'ggH125_ZZ4lep']
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
        hist_simulation = axes[0].hist(events,
                                       weights=weights,
                                       bins=variable['binning'],
                                       label=labels,
                                       color=colors,
                                       stacked=True)
    else:
        hist_simulation = axes[0].hist(events,
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
        bin_center = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        axes[0].errorbar(
            bin_center,
            data_num,
            yerr=np.sqrt(data_num),
            color='k',
            label='data',
            ls='none',
            marker='.',
        )
        # Ratio plot
        axes[1].errorbar(
            bin_center,
            array_division(data_num, hist_simulation[0][-1]),
            yerr=array_division(np.sqrt(data_num), hist_simulation[0][-1]),
            color='k',
            ls='none',
            marker='.',
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
    return fig, axes


def plot_normed_signal_vs_background(variable: Dict[str, Union[str, List, Dict]], data_frame_signal: pd.DataFrame,
                                     data_frame_background: pd.DataFrame) -> Tuple[figure.Figure, List[plt.Axes]]:
    """Plots the normalized signal and background distribution for a given variable.

    Args:
        variable (dict): A dictionary containing information about the variable to be plotted.
        data_frame_signal (pd.DataFrame): Dataframe containing signal data.
        data_frame_background (pd.DataFrame): Dataframe containing background data.

    Returns:
        Tuple[figure.Figure, List[plt.Axes]]: A tuple containing the figure and list of axes.
    """
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
    non_overlap = sum(abs(hist_signal[0] - hist_background[0])) / 2.
    axes[1].text(0.01, 0.95, 'Non-overlap = {}'.format(np.round(non_overlap, 4)),
                 verticalalignment='top', transform=axes[1].transAxes)
    return fig, axes
