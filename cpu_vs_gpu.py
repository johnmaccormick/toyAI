# written by Colab AI, minor jmac adjustments
# prompt: Please write some pytorch code that demonstrates the difference in training time for a moderately sized neural network when run on a GPU compared to a CPU.

import torch
import torch.nn as nn
import torch.optim as optim
import time

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define a moderately sized neural network


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1000, 500)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(500, 250)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(250, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


# Generate some random data
input_size = 1000
num_samples = 10000
inputs = torch.randn(num_samples, input_size)
targets = torch.randint(0, 10, (num_samples,))

# Create the model and optimizer
model = Net()
criterion = nn.CrossEntropyLoss()

# Function to train the model


def train(model, device, inputs, targets, epochs=50):
    model.to(device)
    optimizer = optim.Adam(model.parameters())
    start_time = time.time()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(inputs.to(device))
        loss = criterion(outputs, targets.to(device))
        loss.backward()
        optimizer.step()
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training time: {training_time:.2f} seconds")
    return training_time


# Train on CPU
print("Training on CPU...")
cpu_training_time = train(model, "cpu", inputs, targets)

# Train on GPU (if available)
if torch.cuda.is_available():
    print("Training on GPU...")
    gpu_training_time = train(model, "cuda", inputs, targets)
    print(f"GPU training was {
          cpu_training_time/gpu_training_time:.2f} times faster than CPU training.")
else:
    print("No GPU available")
