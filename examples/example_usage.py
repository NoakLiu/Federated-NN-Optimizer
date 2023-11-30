from nn_optimizer.mlp import MLP
import numpy as np


def main():
    # Example usage of the MLP
    # Assuming you have a dataset loaded in X_train, y_train, X_test, y_test
    X_train, y_train, X_test, y_test = load_your_data()  # replace with your data loading method
    n_input = X_train.shape[1]
    n_classes = len(np.unique(y_train))
    layers = [50, 70]  # Example layer structure

    # Initialize MLP
    nn = MLP(n_input, n_classes, layers, optimizer='SGD', activation='relu', activation_last_layer='softmax',
             dropout_ratio=0, is_batch_normalization=False)

    # Train the model
    nn.fit(X_train, y_train, epochs=100, learning_rate=0.01)

    # Evaluate the model
    predictions = nn.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")


def load_your_data():
    # Dummy function, replace with actual data loading
    train_data = np.load("dataset/train_data.npy", encoding='bytes')
    train_label = np.load("dataset/train_label.npy", encoding='bytes')
    test_data = np.load("dataset/test_data.npy", encoding='bytes')
    test_label = np.load("dataset/test_label.npy", encoding='bytes')
    train_dataset = np.hstack([train_data, train_label])
    test_dataset = np.hstack([test_data, test_label])
    return train_data, train_label, test_data, test_label


if __name__ == "__main__":
    main()
