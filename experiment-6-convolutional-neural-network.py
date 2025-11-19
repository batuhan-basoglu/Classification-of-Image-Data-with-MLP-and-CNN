import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

class CNN:
    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._build_model().to(self.device)
        self.criterion = nn.CrossEntropyLoss() # cross entropy is used for loss from torch
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001) # adam is used for optimization from torch

    def _build_model(self):
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(1, 32, 3, padding=1) # convolutional layer 1
                self.conv2 = nn.Conv2d(32, 64, 3, padding=1) # convolutional layer 2
                self.fc1 = nn.Linear(64 * 7 * 7, 256) # hidden layer with 256 units
                self.fc2 = nn.Linear(256, 10) # output layer
                self.relu = nn.ReLU() # ReLU activation from torch
                self.pool = nn.MaxPool2d(2, 2) # pooling from torch

            def forward(self, x):
                # forwards pass through the network
                x = self.relu(self.conv1(x))
                x = self.pool(x)
                x = self.relu(self.conv2(x))
                x = self.pool(x)
                x = x.view(x.size(0), -1)
                x = self.relu(self.fc1(x))
                x = self.fc2(x)
                return x

        return Net()

    def fit(self, train_loader, val_loader, epochs=10):
        self.train_losses = []
        self.val_accuracies = []

        for epoch in range(1, epochs + 1):
            self.model.train()
            epoch_loss = 0.0

            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() # updating the epoch loss

            self.train_losses.append(epoch_loss)

            val_acc = self.evaluate(val_loader)
            self.val_accuracies.append(val_acc)
            print(f"Epoch {epoch:02d} | Training Loss: {epoch_loss:.4f} | Validation Accuracy: {val_acc:.4f}")

        self.plot_graph(self.train_losses, self.val_accuracies)
        return self.val_accuracies[-1]

    def evaluate(self, loader):
        # measures the accuracy of your model on a given dataset
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return correct / total

    def plot_graph(self, train_losses, val_accuracies):
        if not os.path.exists('results'):
            os.makedirs('results') # creates results director

        fig, ax1 = plt.subplots() # initializes the plot

        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Training Loss', color='tab:blue')
        ax1.plot(range(1, len(train_losses)+1), train_losses, color='tab:blue', label='Training Loss')
        ax1.tick_params(axis='y', labelcolor='tab:blue') # defines loss subplot

        ax2 = ax1.twinx()
        ax2.set_ylabel('Validation Accuracy', color='tab:orange')
        ax2.plot(range(1, len(val_accuracies)+1), val_accuracies, color='tab:orange', label='Validation Accuracy')
        ax2.tick_params(axis='y', labelcolor='tab:orange') # defines accuracy subplot

        plt.title('CNN Training Loss and Validation Accuracy over Epochs')

        result_path = 'results/experiment-6-convolutional-neural-network.png' # defines the file name
        fig.savefig(result_path)
        print(f"Graph saved to: {result_path}")

    def predict(self, loader):
        # returns the predicted class labels for all samples in the loader, instead of summarizing them as accuracy.
        self.model.eval()
        all_preds = []
        with torch.no_grad():
            for images, _ in loader:
                images = images.to(self.device)
                outputs = self.model(images)
                preds = torch.argmax(outputs, dim=1)
                all_preds.append(preds.cpu())
        return torch.cat(all_preds)

# pre-processing the images
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# acquiring the FashionMNIST dataset
train_set = datasets.FashionMNIST(root='.', train=True, download=True, transform=transform)
test_set = datasets.FashionMNIST(root='.', train=False, download=True, transform=transform)

# splits data using dataloader with 256 batches
train_loader = DataLoader(train_set, batch_size=256, shuffle=True)
test_loader = DataLoader(test_set, batch_size=256, shuffle=False)

# CNN initialization and training the model
cnn = CNN()
cnn.fit(train_loader, test_loader, epochs=10)

# tests the model
test_acc = cnn.evaluate(test_loader)
print(f"\nFinal test accuracy: {test_acc:.4f}")