# Retina Nerve Segmentation: Preventing Blindness with Deep Learning

Today, there are 95 million people with Diabetic Retinopathy worldwide. This disease affects the retina blood vessels, having caused blindness for 850,000 people leaving today. In order to improve prevention and treatment, we want to develop a Retina Nerve Segmentation algorithm.

The goal of this tutorial is to illustrate the use of Deep Learning for applications in Medical Imaging. After completing this tutorial, you will be able to segment nerves in retina images with a Convolutional Neural Network. This tutorial is destined to people who already have some notions of coding in Python and some notions of machine learning.

The proposed Network shows an **AUC of 0.9708** on the first test image of the DRIVE dataset.

## How to

**Download the dataset**. Go to http://www.isi.uu.nl/Research/Databases/DRIVE/download.php and follow the instructions to download the DRIVE folder. In this folder, take the "training" and "test" subfolders and put them in the empty "data" folder of this repository.

**Install Jupyter**. Jupyter is automatically available if you install Anaconda (https://www.continuum.io/downloads).

**Install the libraries**. The libraries we will use are matplotlib.pyplot, numpy, os, matplotlib.image, keras, theano, sklearn, pandas, skipy.misc and pickle.

**Run the code**. Follow the Jupyter notebook main_commented.ipynb.
