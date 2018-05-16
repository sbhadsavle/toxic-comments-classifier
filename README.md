# Toxic Comment Classification to detect Threats, Obscenity, Insults, and Identity-Based Hate

## Overview

"Toxic comments" are disrespectful or unreasonable comments that are likely to make a participant leave a discussion. Because these sorts of comments have traditionally been difficult to detect automatically in online platforms, we present a novel classification system that solves this problem by categorizing comments as: toxic, severely toxic, threatening, obscene, insulting, and hateful. Our approach draws upon past work from fields like sentiment analysis, textual classification, and toxic behavior detection. The novelty of our work in comparison to past research lies in the ability of our system to effectively detect different forms of toxicity as opposed to a single sentiment or toxicity metric. We select a linear classifier as our baseline and improve upon it by performing additional feature extraction and by training alternative classifiers, including an ensemble and a neural network. Upon experimentation, we see that our approach is more accurate across all 6 categories of toxicity and also shows almost a 0.20 point improvement in the general "Toxic" category. Lastly, we develop a web tool using our toxic classification system that performs on-par if not better than a prominent toxicity detection web tool that is currently available.

This work was developed as part of a final course project for INF385T: Introduction to Machine Learning at the University of Texas at Austin in Spring 2018. The premise of this work is from [this](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) Kaggle competition.

## Dataset

We use the Wikipedia comments dataset provided by Kaggle [here](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data). It consists of 159,571 instances categorized by humans to be toxic, severely toxic, threatening, obscene, insulting, and hateful.

## Baseline Kernel

We use a [baseline](https://www.kaggle.com/jhoward/nb-svm-strong-linear-baseline) developed by Jeremy Howard, which was submitted to the Kaggle competition.

## Augmented Kernel

We improve upon the baseline by augmenting the training data with toxicity values and sentiment values using Google Perspective API and Microsoft Azure Text Analytics API. This feature extraction procedure took approximately 7 hours for all training instances on a NC6 Microsoft Azure data science virtual machine with 6 VCPUs, 56 GB of RAM, and a Tesla K80 GPU. We also substitute the logistic regression classifier with a random forest classifier to see what improvements we could acheive.

## Neural Kernel

Using the same strategy as the Augmented Kernel, we use the Keras deep learning library to try a simple neural network as the classifier. The network consists of an input layer, a densely connected layer with tanh as the activation function, a 15% dropout layer, and a final densely connected layer with relu as the activation function. The training of this neural network consisted of iterating over the entire dataset over 3 epochs using the Nadam optimizer.
