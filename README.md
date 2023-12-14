

# NN Optimizer

This project implements a neural network optimizer with various features like activation functions, dropout, batch normalization, and multiple optimizers.

## Installation

Clone the repository and install dependencies.

```bash
git clone <repository-url>
cd federated-nn-optimizer
pip install -r requirements.txt
```

### Docker

If you have Docker installed, you can use it to build and run the NN Optimizer project:

1. Build the Docker image:

```bash
docker build -t federated-nn-optimizer .
```

2. Run the Docker container:

```bash
docker run federated-nn-optimizer
```

This will execute the example usage script in a Docker container.

## Usage

Refer to the `examples/example_usage.py` file for an example of how to use the NN Optimizer package. This script demonstrates initializing the MLP model, training it with your data, and making predictions.

## Features

- Activation Functions: Includes ReLU, Sigmoid, Tanh, Softmax, etc.
- Dropout: Provides dropout functionality for neural network regularization.
- Batch Normalization: Helps to speed up training and improve performance.
- Optimizers: Supports various optimization algorithms like SGD, Momentum, NAG, and Adagrad.

## Discussion

Discussion to the NN Optimizer project are welcome. You can contact me by dliu328@wisc.edu
