# Electroencephalography (EEG) Signal Classification using Deep Learning

In this project, I use a Convolutional Neural Network to accomplish a task in the field of [Brain Computer Interface (BCI)](https://en.wikipedia.org/wiki/Brain%E2%80%93computer_interface) called *[P300](https://en.wikipedia.org/wiki/P300_(neuroscience)) speller paradigm*. My goal was to tell to which letter on the speller the subject was paying attention through P300 (a kind of deflection in EEG caused by certain stimuli).

The accuracy of this model is **92%** (for subject A), which can rank **second** in this contest.

![P300](https://3c1703fe8d.site.internapcdn.net/newman/csz/news/800/2015/562df18a48c5c.png)

## Data
Data set II: ‹P300 speller paradigm› from [BCI Competition III](http://www.bbci.de/competition/iii/).

You can read more about this data set [here (pdf)](http://www.bbci.de/competition/iii/desc_II.pdf).

## Jupyter Notebook
You can read through the entire process in the Jupyter Notebook [here](https://github.com/Yuchen-Wang-SH/Electroencephalography-EEG-Signal-Classification-using-Deep-Learning/blob/master/notebooks/Electroencephalography%20(EEG)%20Signal%20Classification%20using%20Deep%20Learning.ipynb).

## Scripts
The scripts used to preprocess the data and construct the model can be found in the [src/ folder](https://github.com/Yuchen-Wang-SH/Electroencephalography-EEG-Signal-Classification-using-Deep-Learning/tree/master/src).

## Want to run locally?
1. First, clone this repo to your computer.
2. Then, you need to go [here](http://www.bbci.de/competition/iii/), fill your information below **Download of data sets**. You will receive an email with your account and password to download their data sets. Remember you should download Data set II.
3. You should create a path of `/data/raw` in the project folder, download and unzip the data set in `raw`, and the path in the Jupyter Notebook should be fine.
4. Get the true labels [here](http://www.bbci.de/competition/iii/results/index.html). In the notebook I refer to the true labels in a text file called `A_test_labels.txt`.
5. Type in command line:

    ```
    pip install -r requirements.txt
    ```
    or:
    ```
    make requirements
    ```
6. You are all set!