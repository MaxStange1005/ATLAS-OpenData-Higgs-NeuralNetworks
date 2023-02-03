#!/usr/bin/env python
# coding: utf-8

# # Discover the Higgs with Deep Neural Networks

# The input data was created from 13 TeV ATLAS open data downloaded from http://opendata.atlas.cern/release/2020/documentation/index.html
# 
# For more information read:<br>
# Review of the 13 TeV ATLAS Open Data release, Techn. Ber., All figures including auxiliary figures are available at https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PUBNOTES/ATL-OREACH-PUB-2020-001: CERN, 2020, url: http://cds.cern.ch/record/2707171

# ## Data Preparation

# ### Load the Data

# In[1]:


# Necessary imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy.random import seed
import common


# The goal of this lab course is to train a deep neural network to separate Higgs boson signal from background events. The most important signal sample ggH125_ZZ4lep corresponds to the process gg->H->ZZ. The dominant background sample is llll resulting from Z and ZZ decays.
# After training the DNN model will be used to classify the events of the data samples.
# 
# Higgs signal samples:
# - ggH125_ZZ4lep
# - VBFH125_ZZ4lep
# - WH125_ZZ4lep
# - ZH125_ZZ4lep
# 
# Background samples:
# - llll
# - Zee
# - Zmumu
# - ttbar_lep
# 
# Data samples:
# - data_A
# - data_B
# - data_C
# - data_D

# In[2]:


# Define the input samples
sample_list_signal = ['ggH125_ZZ4lep']
sample_list_background = ['llll']
sample_list_signal = ['ggH125_ZZ4lep', 'VBFH125_ZZ4lep', 'WH125_ZZ4lep', 'ZH125_ZZ4lep']
sample_list_background = ['llll', 'Zee', 'Zmumu', 'ttbar_lep']


# In[3]:


# Read all the samples
no_selection_data_frames = {}
for sample in sample_list_signal + sample_list_background:
    no_selection_data_frames[sample] = pd.read_csv('input/' + sample + ".csv")


# ### Input Variables

# The input provides several variables to classify the events. Since each event has multiple leptons, they were ordered in descending order based on their transverse momentum. Thus, lepton 1 has the highest transverse momentum, lepton 2 the second highest, and so on. <br>
# Most of the given variables can be called low-level, because they represent event or object properties, which can be derived directly from the reconstruction in the detector. In contrast to this are high-level variables, which result from the combination of several low-level variables. In the given dataset the only high-level variables are invariant masses of multiple particles:<br>
# $m_{inv} = \sqrt{\left(\sum\limits_{i=1}^{n} E_i\right)^2 - \left(\sum\limits_{i=1}^{n} \vec{p}_i\right)^2}$
# 
# 
# List of all available variables:<br>
# - Event number
#  - Each simulated event has its own specific number. (not used for training)
#  - Variable name: eventNumber
# - Scale and event weight
#  - The scaling for a data set is given by the sum of event weights, the cross section, luminosity and a efficiency scale factor
#  - Each event has an additional specific event weight
#  - To combine simulated events and finally compare them to data each event has to be scaled by the total weight, the product of scale weight and event weight
#  - The weight are not used for training
#  - Variable name: scaleWeight, eventWeight, totalWeight
# - Number of jets
#  - Jets are particle showers which result primarily from quarks and gluons
#  - Variable name: jet_n
# - Invariant four lepton mass
#  - The invariant mass $m_{inv}(l_1, l_2, l_3, l_4)$ of the four leptons is extremly sensitive to Higgs boson events. This variable is to be displayed later and thus not used for training
#  - Variable name: lep_m_llll
# - Invariant two lepton mass
#  - Invariant masses $m_{inv}(l_i, l_j)$ of all combinations of two leptons
#  - Variable names: lep_m_ll_12, lep_m_ll_13, lep_m_ll_14, lep_m_ll_23, lep_m_ll_24, lep_m_ll_34
# - Transverse momentum of the leptons
#  - The momentum in the plane transverse to the beam axis
#  - Variable names: lep1_pt, lep2_pt, lep3_pt, lep4_pt
# - Lepton azimuthal angle
#  - The azimuthal angle $\phi$ is measured in the plane transverse to the beam axis
#  - Variable name: lep1_phi, lep2_phi, lep3_phi, lep4_phi
# - Lepton pseudo rapidity
#  - The angle $\theta$ is measured between the lepton track and the beam axis. Since this angle is not invariant against boosts along the beam axis, the pseudo rapidity $\eta = - \ln{\tan{\frac{\theta}{2}}}$ is primarily used in the ATLAS analyses
#  - Variable names: lep1_eta, lep2_eta, lep3_eta, lep4_eta
# - Lepton energy
#  - The energy of the leptons reconstructed from the calorimeter entries
#  - Variable name:lep1_e, lep2_e, lep3_e, lep4_e
# - Lepton PDG-ID
#  - The lepton type is classified by a n umber given by the Particle-Data-Group. The lepton types are $pdg-id(e)=11$, $pdg-id(\mu)=13$ and $pdg-id(\tau)=15$
#  - Variable name: lep1_pdgId, lep2_pdgId, lep3_pdgId, lep4_pdgId
# - Lepton charge
#  - The charge of the given lepton reconstructed by the lepton track
#  - Variable name: lep1_charge, lep2_charge, lep3_charge, lep4_charge

# ### Event Pre-Selection

# Although the final selection of the data is to be performed on the basis of a DNN, a rough pre-selection of the data is still useful.
# For this purpose, selection criteria are defined, which return either true or false based on the event kinematics and thus decide whether the respective event is kept or discarded.
# Suitable criteria for this analysis are very basic selections that must be clearly fulfilled by H->ZZ->llll processes.
# 
# 
# Hint: What lepton types and charges are expected in the final state?

# In[4]:


def cut_lep_type(lep_type_0, lep_type_1, lep_type_2, lep_type_3):
    # Only keep events like eeee, mumumumu or eemumu
    sum_lep_type = lep_type_0 + lep_type_1 + lep_type_2 + lep_type_3
    return sum_lep_type == 44 or sum_lep_type == 48 or sum_lep_type == 52


def cut_lep_charge(lep_charge_0, lep_charge_1, lep_charge_2, lep_charge_3):
    # Only keep events where the sum of all lepton charges is zero
    sum_lep_charge = lep_charge_0 + lep_charge_1 + lep_charge_2 + lep_charge_3
    return sum_lep_charge == 0


# In[5]:


# Create a copy of the original data frame to investigate later
data_frames = no_selection_data_frames.copy()

# Apply the chosen selection criteria
for sample in sample_list_signal + sample_list_background:
    # Selection on lepton type
    type_selection = np.vectorize(cut_lep_type)(
        data_frames[sample].lep1_pdgId,
        data_frames[sample].lep2_pdgId,
        data_frames[sample].lep3_pdgId,
        data_frames[sample].lep4_pdgId)
    data_frames[sample] = data_frames[sample][type_selection]

    # Selection on lepton charge
    charge_selection = np.vectorize(cut_lep_charge)(
        data_frames[sample].lep1_charge,
        data_frames[sample].lep2_charge,
        data_frames[sample].lep3_charge,
        data_frames[sample].lep4_charge)
    data_frames[sample] = data_frames[sample][charge_selection]


# ### Data Investigation

# Before one can decide which variables are suitable for training, one must first get a feel for the input variables.
# For this purpose, the input samples are merged into a set of signal events and a set of background events. Afterwards, the behavior of signal and background can be studied in multiple variables.

# In[6]:


# Merge the signal and background data frames
def merge_data_frames(sample_list, data_frames_dic):
    for sample in sample_list:
        if sample == sample_list[0]:
            output_data_frame = data_frames_dic[sample]
        else:
            output_data_frame = pd.concat([output_data_frame, data_frames_dic[sample]], axis=0)
    return output_data_frame

data_frame_signal = merge_data_frames(sample_list_signal, data_frames)
data_frame_background = merge_data_frames(sample_list_background, data_frames)


# The function common.plot_hist(variable, data_frame_1, data_frame_2) plots the given variable of the two data sets.
# The variable must be a dictionary containing atleast the variable to plot. Additionally one can also specify the binning (list or numpy array) and the xlabel. The created histogram is automatically saved in the plots directory<br>
# An example for the transverse momnetum of the leading lepton is given below:

# In[7]:


# leading lepton pt
var_lep1_pt = {'variable': 'lep1_pt',
               'binning': np.linspace(0, 300, 100),
               'xlabel': '$p_T$ (lep 1) [GeV]'}

common.plot_hist_sum(var_lep1_pt, data_frames)
common.plot_normed_signal_vs_background(var_lep1_pt, data_frame_signal, data_frame_background)


# Investigate the signal vs background ratio before the training

# In[8]:


signal_event_number = sum(data_frame_signal.totalWeight)
background_event_number = sum(data_frame_background.totalWeight)
signal_background_ratio = signal_event_number/background_event_number
print(len(data_frame_signal.totalWeight), len(data_frame_background.totalWeight))
print(f'There are {round(signal_event_number, 2)} signal and {round(background_event_number, 2)} backgound events\n This gives us a purity of {round(signal_background_ratio*100, 2)}%')


# ## Training

# In this chapter, the DNN is trained. Define the input variables for the classification. Feel free to test different combinations of low-level and high-level variables. Additionally, you can modify the structure of the DNN to gain optimal training results. When the setup is done, train the DNN, test it and redo this until you are happy with the result.

# In[9]:


# The training input variables
training_variables = ['lep1_pt', 'lep2_pt', 'lep3_pt', 'lep4_pt']
training_variables += ['lep1_charge', 'lep2_charge', 'lep3_charge', 'lep4_charge']
training_variables += ['lep1_pdgId', 'lep2_pdgId', 'lep3_pdgId', 'lep4_pdgId']
training_variables += ['lep1_phi', 'lep2_phi', 'lep3_phi', 'lep4_phi']
training_variables += ['lep1_eta', 'lep2_eta', 'lep3_eta', 'lep4_eta']
#training_variables += ['lep_m_ll_12', 'lep_m_ll_13', 'lep_m_ll_14', 'lep_m_ll_23', 'lep_m_ll_24', 'lep_m_ll_34']


# In[10]:


# Create the training input
input_values = []
input_weights = []
input_classification = []
for sample in sample_list_signal + sample_list_background:
    # Classify signal and background (and skip if data)
    if sample in sample_list_signal:
        # 1 if signal
        input_classification.append(np.ones(len(data_frames[sample])))
    elif sample in sample_list_background:
        # 0 if background
        input_classification.append(np.zeros(len(data_frames[sample])))
    else:
        continue
    # input values
    input_values.append(data_frames[sample][training_variables])
    input_weights.append(data_frames[sample]['totalWeight'])

# Merge the input
input_values = np.concatenate(input_values)
input_weights = np.concatenate(input_weights)
input_classification = np.concatenate(input_classification)


# The data set is split in a training, test and validation set. To have enough data to train use a splitting of 80%:10%:10%

# In[11]:


from sklearn.model_selection import train_test_split


# In[12]:


train_input_values, test_input_values, train_input_classification, test_input_classification = train_test_split(input_values, input_classification, test_size=0.2, random_state=420)

# Reweight signal and background
train_weights = train_test_split(input_weights, input_classification, test_size=0.2, random_state=420)[0]

train_weights_signal = train_weights*train_input_classification
train_weights_background = train_weights*(1 - train_input_classification)

print('All training events:', sum(train_weights))
print('training signal events:', sum(train_weights_signal))
print('training background events:', sum(train_weights_background))

# Reweight the signal events
weight_correction = sum(train_weights_background) / sum(train_weights_signal)
train_weights_1to1 = train_weights_background + train_weights_signal * weight_correction

print('After weight correction:')
print('All training events:', sum(train_weights_1to1))
print('training signal events:', sum(train_weights_1to1*train_input_classification))
print('training background events:', sum(train_weights_1to1*(1 - train_input_classification)))


# In[13]:


# Import the tensorflow module to create a DNN

import tensorflow as tf
import pandas as pd


# Create you DNN consisting of several layers. 
# 
# A full list of layer types can be found here https://www.tensorflow.org/api_docs/python/tf/keras/layers <br>
# Some important examples for layers are:
# - Dense: a densely connected NN layer
# - Flatten: flattens the input
# - Dropout: randomly sets input to 0. Can decrease overtraining
# 
# The neurons of each layer are activated by the so called activation function. <br>
# A full list of provided activation functions can be found here https://www.tensorflow.org/api_docs/python/tf/keras/activations <br>
# Some examples are:
# - linear: $linear(x) = x$
# - relu: linear for $x>0$ ($relu(x) = max(0, x)$)
# - exponential: $exponential(x) = e^x$
# - sigmoid: $sigmoid(x) = 1 / (1 + e^{-x})$
# 
# To classify background and signal the last layer should only consist of one neuron which has an activation between 0 and 1

# <b> SCALE INPUT VARIABLES! (with sklearn) </b>
# 
# What about the initialization of the weights?

# In[14]:


n_layers = [2, 4, 6, 8]
n_nodes = [5, 10, 15, 20, 30, 40]
model = tf.keras.models.Sequential([
  tf.keras.layers.Normalization(mean=0, variance=1),
  tf.keras.layers.Dense(40, activation='relu'),
  tf.keras.layers.Dense(40, activation='relu'),
  tf.keras.layers.Dense(40, activation='relu'),
  tf.keras.layers.Dense(40, activation='relu'),
  tf.keras.layers.Dense(1, activation='sigmoid')
])


# While training, the parameters of the neural network are modified to increase the agreement between the classification of the training data by the neural network and the actual splitting in signal and background. 
# The agreement is specified by a loss-function, which decreses with increasing agreement. Analogous to a Chi2 fit, the fit to the training data is done by determining the minimum of the loss-function. The choice of the specific loss-function has to be adapted to the problem the neural network has to solve. <br>
# 
# The list of available loss-functions are listed here https://www.tensorflow.org/api_docs/python/tf/keras/losses
# For the binary (0 or 1) classification problem of the signal background separation the BinaryCrossentropy is used. <br>
# <b> Give more information!!! </b>

# In[15]:


loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)  # verify if correct


# For the optimization of the loss-function different optimization algorithms can be used (see https://www.tensorflow.org/api_docs/python/tf/keras/optimizers). <br>
# The performance of the learning process is given by the metric (see https://www.tensorflow.org/api_docs/python/tf/keras/metrics)

# In[16]:


model.compile(optimizer='adam',
              loss=loss_fn,
              weighted_metrics=['binary_accuracy'])


# early stopping?

# In[19]:


pd.Series(train_weights)


# In[20]:


model.fit(train_input_values, train_input_classification, sample_weight=pd.Series(train_weights), epochs=2) #


# In[ ]:


model.evaluate(train_input_values,  train_input_classification, verbose=2)


# In[ ]:


model.evaluate(test_input_values,  test_input_classification, verbose=2)


# ## Test the Model

# In[ ]:


# Prediction for test sample
test_prediction = model.predict(test_input_values)


# In[ ]:



def calculate_roc_curve(prediction, truth_classification):
    # convert prediction if it is a list of lists
    if not any(isinstance(element, float) for element in prediction):
        prediction = [element[0] for element in prediction]
    
    # Loop over a range of different DNN scores
    x_range = np.linspace(1, 0, 100)
    true_positive = []
    false_positive = []
    
    # Numb er of signal and background events
    signal_number = sum(truth_classification)
    background_number = len(truth_classification) - signal_number
    
    # Boolean if signal or background
    bool_signal = truth_classification > 0.5
    bool_background = truth_classification < 0.5
    for x in x_range:
        bool_possitive = prediction > x
        bool_true_positive = bool_possitive & bool_signal
        bool_false_positive = bool_possitive & bool_background
        true_positive.append(bool_true_positive.sum()/signal_number)
        false_positive.append(bool_false_positive.sum()/background_number)

    return true_positive, false_positive

true_positive_rate, false_positive_rate = calculate_roc_curve(test_prediction, test_input_classification)


# In[ ]:


# Area under roc curve
from scipy import integrate

def calc_list_integral(x_list, y_list):
    # Remove redundancy before integration
    no_redundacy_x = []
    no_redundacy_y = []
    for i in range(len(x_list)):
        if x_list[i] not in no_redundacy_x:
            no_redundacy_x.append(x_list[i])
            no_redundacy_y.append(y_list[i])
    integral = integrate.simpson(x = no_redundacy_x, y = no_redundacy_y)
    return integral

print(calc_list_integral(true_positive_rate, false_positive_rate))
print(calc_list_integral(false_positive_rate, true_positive_rate))


# In[ ]:


def plot_roc_values(true_positive_rate, false_positive_rate):
    """This function plots the roc curve for the given values"""
    fig = plt.figure(figsize=(7, 5))
    plt.plot(false_positive_rate, true_positive_rate)
    
    # Style
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.savefig('plots/roc_curve.pdf')
    plt.grid()
    plt.show()

plot_roc_values(true_positive_rate, false_positive_rate)


# ## Apply Model

# In[ ]:


for sample in sample_list_signal + sample_list_background:
    print(f'Apply Model for {sample}')
    # Get the values to apply the model
    values = data_frames[sample][training_variables]
    prediction = model.predict(values)
    
    # Convert prediction to array
    prediction = [element[0] for element in prediction]
    
    # Add the prediction for each sample
    data_frames[sample]['dnn_prediction'] = prediction


# In[ ]:


# Investigate DDN Prediction
var_lep1_pt = {'variable': 'dnn_prediction',
               'binning': np.linspace(0.1, 1, 50),
               'xlabel': 'DNN prediction'}

common.plot_hist_sum(var_lep1_pt, data_frames)


# In[ ]:


# Create a copy of the data frame
apply_dnn_data_frames = data_frames.copy()

# Apply the DNN value
for sample in sample_list_signal + sample_list_background:
    print(sample)
    pass_dnn = apply_dnn_data_frames[sample]['dnn_prediction'] > 0.5
    apply_dnn_data_frames[sample] = apply_dnn_data_frames[sample][pass_dnn]
    print(len(apply_dnn_data_frames[sample])/len(data_frames[sample]))


# In[ ]:


# leading lepton pt
var_lep1_pt = {'variable': 'lep_m_llll',
               'binning': np.linspace(0, 200, 40),
               'xlabel': '$m_{llll}$ [GeV]'}

common.plot_hist_sum(var_lep1_pt, apply_dnn_data_frames)


# In[ ]:




