import torch
import torch.nn as nn
from torch.optim import SGD
from fedpggd import PerGodGradientDescent  # Or other optimizers in the folder federated-nn-optimizer

# Define a simple neural network model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(2, 1)

    def forward(self, x):
        return self.fc(x)

# Create an instance of your custom optimizer (Or other optimizers in the folder federated-nn-optimizer)
model = SimpleModel()
custom_optimizer = PerGodGradientDescent(model.parameters(), learning_rate=0.01, mu=0.01)

# Create a loss function and a sample dataset
criterion = nn.MSELoss()
data = torch.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
target = torch.tensor([3.0, 4.0, 5.0])

# Training loop
for epoch in range(100):
    # Forward pass
    outputs = model(data)
    loss = criterion(outputs, target)

    # Zero the gradients
    custom_optimizer.zero_grad()

    # Backward pass
    loss.backward()

    # Update the model's parameters using your custom optimizer
    custom_optimizer.step()

    # Print the loss for monitoring
    print(f'Epoch [{epoch + 1}/100], Loss: {loss.item()}')
