# Probabilistic Models for Reasoning Object Co-occurrences

Source code for all of the models ran for master project's completion. The code still needs work to document and tidy as some implementation code is quite messy and needed to add some comments on each functionalities.

## Data Source

The original data is the COCO 2017 subset's object detection annotation which we extract the object counts per image.

## Python Implementation

### Setup

To setup for running the Python code, please install the required packages in requirements.txt within the `co-occurrence-model` folder.

```
pip install -r requirements.txt
```

### Model Training and Testing

To train and test the models that we used, please refer to the existing notebooks and the .sh script on ways to train and test the models. einet-em-Copy1.sh can be ignored since it is only used for sanity checks. To see the hyperparameters used for training, refer to the `train*.py` files.

## MATLAB Implementation

For BIN-G, please follow the Readme file in the folder. To reproduce our result, open the evaluate_coco2017.m in MATLAB and run it.
