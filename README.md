# Boiling Heat Flux Prediction via Temporal-Spatial Models
This repository consists of code for generating and testing models for predicting boiling heat flux from image data from transient pool boiling experiments. It consists of different types of machine learning models including x,y,z.  

This github is from the work presented [here](https://ieeexplore.ieee.org/document/10680575).
## Organization of Repository:
Consists of mulitple folders as described:
* DataGeneration- Code for preparing data into sequences. The goal of these scripts is so that images are not needed to be loaded into memory. It also matches each image to a heat flux calculated from thermocouple measurements.
* Hyperparameter Tuning- Code for selecting the best hyperparameters for each model.
* ModelTraining- Code for training each best model as determined by hyperparameter tuning. Best weights are then saved for testing.
* TestingModels- Codes for testing each model on unseen test data
* Analysis- Code for generating plots from trained models. 
## Preliminary Setup:
1. Clone github repo
2. Download pretrained models here:
