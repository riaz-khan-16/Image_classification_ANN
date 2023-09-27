# Task2 Image_classification_ANN


## Explanation of the configuration and training process:
## Data Loading and Preprocessing:
● We load the MNIST dataset using mnist.load_data().
● We normalize the pixel values in the range [0, 1] by dividing by 255.
● We one-hot encode the labels using to_categorical.

## Neural Network Architecture:
● We create a sequential model.
● The input layer is a Flatten layer that converts the 28x28 input images into a flat
vector.
● We add a hidden layer with 128 units and ReLU activation.
● The output layer has 10 units (one for each digit) with softmax activation.
Model Compilation:
● We use the Adam optimizer, a popular choice for gradient-based optimization.
● Categorical cross-entropy is used as the loss function since this is a classification
task.
● We monitor the accuracy metric during training.

## Training:
● We train the model on the training data for 5 epochs with a batch size of 64.
● We use a validation split of 20% to monitor the model's performance during training.
Evaluation:
● We evaluate the model on the test dataset to obtain accuracy and loss metrics.
This basic feedforward neural network should provide decent accuracy on the MNIST
dataset. You can experiment with different architectures, optimizers, and hyperparameters to
further improve the performance. Additionally, you can visualize the training history to
observe how accuracy and loss change over epochs to help with model tuning.
Model Configuration:
● Input Layer: Flatten layer (28x28 to a flat vector)
● Hidden Layer: Dense layer with 128 units and ReLU activation
● Output Layer: Dense layer with 10 units (one for each digit) and softmax activation
● Optimizer: Adam
● Loss Function: Categorical Cross-Entropy
● Metrics: Accuracy
●
## Training Details:
● Number of Epochs: 5
● Batch Size: 64
● Validation Split: 20% of the training data
## Training Results:
● Training Accuracy (Final Epoch): 0.9787
● Training Loss (Final Epoch): 0.0752
● Validation Accuracy (Final Epoch): 0.9753
● Validation Loss (Final Epoch): 0.0882
## Test Results:
● Test Accuracy: 0.9742
● Test Loss: 0.0887
