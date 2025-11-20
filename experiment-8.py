import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_class = 10
batch_size = 256
epochs = 10
lr = 2e-3
num_workers = os.cpu_count()
step_size = 5
gamma = 0.1

transform_train = transforms.Compose([
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.Grayscale(num_output_channels=3),  # MobileNet expects 3 channels
    transforms.Resize((28, 28)),  # Resize the input images to 28x28
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((28, 28)),  # Resize the test images to 28x28
    transforms.ToTensor(),
])

train_set = datasets.FashionMNIST(root='.', train=True, download=True, transform=transform_train)
test_set = datasets.FashionMNIST(root='.', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)

# freezes convolutional layers
for param in model.features.parameters():
    param.requires_grad = False

# function to get the size of the features after convolutional layers
def get_fc_input_size(model, input_size=(3, 28, 28)):
    batch_size = 16  # uses a small batch size to avoid issues with batch normalization
    x = torch.randn(batch_size, *input_size).to(device)  # Batch size of 16
    # passes through the model up to the classifier to get the feature map size
    with torch.no_grad():
        features = model.features(x)
    # flattens the feature map to calculate the input size for the first fully connected layer
    return features.view(batch_size, -1).size(1)

# calculates the correct input size for the first FC layer
fc_input_size = get_fc_input_size(model, input_size=(3, 28, 28))

# replaces the classifier with new fully connected layers
model.classifier = nn.Sequential(
    nn.Linear(fc_input_size, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, num_class)
)

# compiles for faster CPU/GPU execution
model = torch.compile(model)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

def plot_graph(train_losses, val_accuracies):
    if not os.path.exists('results'):
        os.makedirs('results')
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Training Loss', color='tab:blue')
    ax1.plot(range(1, len(train_losses)+1), train_losses, color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Validation Accuracy', color='tab:orange')
    ax2.plot(range(1, len(val_accuracies)+1), val_accuracies, color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    plt.title('Training Loss and Validation Accuracy')
    fig.savefig('results/experiment-8.png')
    print("Graph saved to results/experiment-8.png")

def training(model, loader, optimizer, criterion):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16 if device.type=='cpu' else torch.float16):
            logits = model(images)
            loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total

@torch.no_grad()
def evaluation(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16 if device.type=='cpu' else torch.float16):
            logits = model(images)
            loss = criterion(logits, labels)
        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total

train_losses, val_accuracies = [], []

for epoch in range(1, epochs + 1):
    train_loss, train_acc = training(model, train_loader, optimizer, criterion)
    val_loss, val_acc = evaluation(model, test_loader, criterion)
    train_losses.append(train_loss)
    val_accuracies.append(val_acc)
    print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")
    scheduler.step()

plot_graph(train_losses, val_accuracies)