# Hunt for the Higgs Boson with Neural Networks
This is a jupyter notebook course introducing to Higgs analysis in combination with neural networks.
The aim is to be able to understand analysis searching for signal contributions by machine learning.

The input data was created by 13 TeV ATLAS open data available at http://opendata.atlas.cern/release/2020/documentation/index.html

For more information read:<br>
Review of the 13 TeV ATLAS Open Data release, Techn. Ber., CERN, 2020, url: http://cds.cern.ch/record/2707171

## Set Up the Environment

We use Anaconda to get the correct environment and all necessary packages. Follow these
steps to get the correct environment and test it:
1. Install Anaconda:<br>
https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html
2. Import the environment environment.yml<br>
Within the shell use `conda env create -n ml_labcourse --file environment.yml`<br>
If you are using a machine with an Apple M1 chip use `conda env create -n ml_labcourse --file environment_mac_m1.yml`
3. Activate the conda environment<br>
`conda activate ml_labcourse`
4. Open a jupyter notebook<br>
`jupyter notebook`
5. Open test_setup.ipynb and execute the cells
If no error occurred you are done :)

## Content
This course is split in several chapters. Since the chapters build on each other and partly also use the previous outputs,
they must be executed in the specified order.

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
- Evaluate and apply the model on training data
- Apply the model on test data to get a theoretical prediction for unseen data

### Chapter 4: Validation Data and Early Stopping
- Introduction to under- and overtraining
- Split in training and validation data
- Early stopping for training
- Evaluation on validation data

### Chapter 5: Training with Event Weights
- Application of weights for training
- Corrections on events weights to enable a successful training

### Chapter 6: Cross-Validation
- Introduction to cross-validation
- Calculate the expected validation loss and its uncertainty

### Chapter 7: Application for Higgs Search
- Introduction to significance calculation
- Calculate significance by cut on model prediction
- Find the best cut value
- Comparison of the models created in chapter 2, chapter 4, and chapter 5

### Chapter 8:
- Now use all low-level variables
- Load baseline model
- Create better model and validate performance via cross-validation
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

## Student Notebook and Solution Notebook
Work in Progress

## TODO:
- How to distribute data?