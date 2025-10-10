# AstroNet-PyTorch
AstroNet translated from TensorFlow to PyTorch

## AstroNet: a neural network for classifying exoplanets transits
Astronet is a deep convolutional neural net (CNN) originally developed in TensorFlow and available [here](https://github.com/tensorflow/models/tree/master/research/astronet). See [Shallue & Vanderburg (2018)](https://arxiv.org/abs/1712.05044) for more information about AstroNet and its application to *Kepler* light curves.

## NASA Frontier Development Lab
In 2018, [NASA's Frontier Development Lab](https://frontierdevelopmentlab.org/) (FDL) formed a team of scientists and machine learning experts to investigate the application of machine learning to detecting and classifying exoplanet transits. As part of this work, the team utilized AstroNet as a baseline model and improved upon it by adding new scientific domain knowledge. 

The 2018 NASA FDL Exoplanet Team:
[Megan Ansdell](www.meganansdell.com),
[Yani Ioannou](https://yani.io/annou/),
[Hugh Osborn](https://www.hughosborn.co.uk/),
[Michele Sasdelli](https://uk.linkedin.com/in/michelesasdelli)

## Astronet-PyTorch
As part of their work in 2018, the NASA FDL Explanet Team translated AstroNet from TensorFlow into PyTorch. Here we make this work publicly available. 
There will soon also be a code for downloading the required *Kepler* light curves and generating the input views and labels; 
for now, you can download the required input files from [this Google Drive link](https://drive.google.com/file/d/1N6bA2rahvV5kcOGmJnTA5gl_Thno7hTh/view?usp=sharing) 
(you must divide them into train, val, and test folders for the code to work)
or follow the instructions on the Astronet GitHub.

If you use this work please cite: [2018 NASA FDL Exoplanet Team (2018), ApJ Letters, 869, L7](http://adsabs.harvard.edu/abs/2018ApJ...869L...7A).

