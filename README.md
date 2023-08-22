# Sentiment Analysis Using IMDB Movie Reviews Data

This is a guide to install and run the training and inference Python scripts provided in this repository for training a sentiment analysis model using the IMDb movie reviews dataset and using this model to predict sentiment of given text. 

## Getting Started

These instructions will guide you on how to execute the scripts on your local machine.

### Prerequisites

This project requires the following python libraries:

* `tensorflow`
* `matplotlib`
* `numpy`

### Installing

Please follow the steps as mentioned below for installing the required libraries.

* Tensorflow
```
pip install tensorflow
```

* Matplotlib
```
pip install matplotlib
```

* Numpy
```
pip install numpy
```

## Running the Code

### Training the Model
* Change the directory to where the training script `train_sentiment_analysis.py` resides.
* Run the file from command prompt/terminal.
```
python train_sentiment_analysis.py
```
* The training script will save the trained model as `model_name.h5`.

### Inference from the Model
* Change the directory to where the inference script `inference_sentiment_analysis.py` resides.
* Run the file from command prompt/terminal after put any movie review.
```
python inference_sentiment_analysis.py
```
* The script will output the sentiment of the given text.

