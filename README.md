

# Federated NN Optimizer

This is a package for federated learning on NN Optimizer, it implements multiple functions of federated NN inlude: avg, aws, dvn, nova, opt, php, reg, rs, pgd, pggd.

## Installation

Clone the repository and install dependencies.

```bash
git clone https://github.com/NoakLiu/Federated-NN-Optimizer.git
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

Refer to the `examples/example_usage.py` file for an example of how to use the Federated NN Optimizer package. This script demonstrates initializing the MLP model, training it with your data on our optimizer models, and making predictions.

## Federated Features

- avg, aws, dvn, nova, opt, php, reg, rs, pgd, pggd

## Reference

- See details in reference folder

## Discussion

Discussion to the Federated NN Optimizer project are welcome. You can contact me by dliu328@wisc.edu
