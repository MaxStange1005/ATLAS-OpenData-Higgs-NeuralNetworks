# Hunt for the Higgs Boson with Neural Networks
This is a jupyter notebook course introducing Higgs to ZZ analysis in combination with neural networks.
The aim is to be able to understand analysis searching for signal contributions by machine learning.

The input data was created from 13 TeV [ATLAS open data](https://opendata.atlas.cern).

For more information read:<br>
Review of the 13 TeV ATLAS Open Data release, Techn. Ber., CERN, 2020, url: http://cds.cern.ch/record/2707171

## Setup the environment

Anaconda/miniforge is used to get the correct environment and all necessary packages. Follow these steps to get the correct environment and test it. In general, Miniforge is recommended since it can speed up the setup process a lot.

Installation with conda:
1. Install Anaconda:<br>
https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html
2. Create the environment:
    - Within the shell use:<br>
    `conda env create -n ml_labcourse --file environment.yml`
    - If you are using a machine with an Apple M1 chip use:<br>
    `conda env create -n ml_labcourse --file environment_mac_m1.yml`

Installation with Miniforge for Mac/Linux:
1. Install Miniforge:<br>
https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html
2. Open a shell and navigate into the neural network course directory
3. Create the environment:
    - Within the shell use:<br>
    `mamba env create -n ml_labcourse --file environment.yml`
    - If you are using a machine with an Apple M1 chip use:<br>
    `mamba env create -n ml_labcourse --file environment_mac_m1.yml`

Installation with Miniforge for Windows:
1. Install Miniforge:<br>
https://github.com/conda-forge/miniforge#miniforge3<br>
Since you are using Windows you probably have to additionally confirm the downloading and installation.
During installation select that you want the Mini Forge Prompt in Windows Start Menu (is probably already selected by default)
2. Start the prompt in your Windows Start Menu as administrator (a shell will open)
3. Navigate into the neural network course directory
4. Create the environment:<br>
`mamba env create --name ml_labcourse -f environment.yml`

#### Test environment

To run and test the notebooks:
1. Mac/Linux:<br>
Open a shell and navigate into the neural network course directory<br>
Windows:<br>
Start your Mini Forge prompt and navigate into the neural network course directory
2. Activate the conda environment:<br>
`conda activate ml_labcourse`<br>
What if the environment cannot be found?
Get the list of available environments<br>
    `conda env list`
    - If there is an environment `/path/to/ml_labcourse` you can directly activate this path<br>
    `conda activate /path/to/ml_labcourse`
3. Open a jupyter notebook:<br>
`jupyter notebook`
4. Open `test_setup.ipynb` and execute the cells

## Input Data

As already mentioned, the input data was created from 13 TeV [ATLAS open data](https://opendata.atlas.cern).

The input data for this course is available as csv files at the TU Dresden Datashare:
https://datashare.tu-dresden.de/s/ZkAgdTyWfHStybw

Download this zip file and unzip it into the directory of the notebooks.

## Content
This course is split into several chapters. Since the chapters build on each other and partly also use the previous outputs, they must be executed in the specified order.

Chapter 0 is recommended for a brief introduction to neural networks. It is independent of the rest of the notebooks and can be used, for example, as preparation for this course.

For chapters 1 to 7, a student notebook and a ready-to-use notebook with the solutions are provided.
Since chapter 8 is usually used as homework for the students its solution is encrypted. As this chapter is primarily about designing your own network, an explicit solution is not very relevant. However, if there is still a need for an "official" solution, ask for the password and decrypt the chapter:

`openssl aes-256-cbc -d -in chapter8_create_your_own_model.ipynb.enc -out chapter8_create_your_own_model.ipynb`


### Chapter 0: A Neural Network with just one Neuron?
This chapter is independent of the following ones and can be used as an introduction to neural networks.
- Create training data
- Build a neural network with just one neuron
- Visualize the training progress

### Chapter 1: Introduction and Data Preparation
- Introduction to the Higgs boson measurement at ATLAS experiment and the golden channel H to ZZ
- Investigation of the input data
- Preselection for the training

### Chapter 2: Create and Train a Neural Network
- Split input in training and test data
- Create tensorflow datasets for the training
- Create a neural network and investigate its architecture
- Train the neural network and plot the training loss
- Save and load the model

### Chapter 3: Evaluate and Apply a Neural Network
- Evaluate and apply the model to training data
- Apply the model to test data to get a theoretical prediction for unseen data

### Chapter 4: Validation Data and Early Stopping
- Introduction to under- and overtraining
- Split in training and validation data
- Early stopping for training
- Evaluation on validation data

### Chapter 5: Training with Event Weights
- Application of weights for training
- Corrections on events weights to enable successful training

### Chapter 6: Cross-Validation
- Introduction to cross-validation
- Calculate the expected validation loss and its uncertainty

### Chapter 7: Application for Higgs Search
- Introduction to significance calculation
- Calculate significance by a cut on model prediction
- Find the best cut value
- Comparison of the models created in chapter 2, chapter 4, and chapter 5

### Chapter 8:
- Now use all low-level variables
- Load baseline model
Create a better model and validate performance via cross-validation
- Calculate Higgs significance
- Compare results to other chapters


## Recommended Prior Knowledge of the Student
- Basics of particle physics
  - Particles of the standard model (leptons, neutrinos, Z boson, Higgs bososn) and their properties
  - Energy, momentum, and charge conservation
  - Reconstructed variables of the ATLAS detector (transverse momentum, invariant mass,...)
- Programming with python
  - Creating functions and loops
  - Knowledge of numpy (required) and maybe also pandas (helpful but not necessary)
  - Jupyter Notebook
- Basic concepts of machine learning