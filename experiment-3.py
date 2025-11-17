import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets
import os

class MLP:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, weight_scale, l1=0, l2=0):
        self.l1 = l1
        self.l2 = l2

        # initializes weights and biases for each layer
        self.W1 = np.random.randn(input_size, hidden_size1) * weight_scale
        self.b1 = np.zeros((1, hidden_size1))
        self.W2 = np.random.randn(hidden_size1, hidden_size2) * weight_scale
        self.b2 = np.zeros((1, hidden_size2))
        self.W3 = np.random.randn(hidden_size2, output_size) * weight_scale
        self.b3 = np.zeros((1, output_size))

    def forward(self, x):
        # forwards pass through the network
        self.x = x  # input for backpropagation
        self.z1 = x @ self.W1 + self.b1  # linear transformation for layer 1
        self.a1 = self.relu(self.z1)  # ReLU activation
        self.z2 = self.a1 @ self.W2 + self.b2  # linear transformation for layer 2
        self.a2 = self.relu(self.z2)  # ReLU activation
        self.z3 = self.a2 @ self.W3 + self.b3  # linear transformation for layer 3
        self.a3 = self.softmax(self.z3)  # applies softmax to get class probabilities
        return self.a3  # output of the network

    def backward(self, y, lr):
        # backwards pass for weight updates using gradient descent
        m = y.shape[0]
        y_one_hot = self.one_hot_encode(y, self.W3.shape[1]) # converts labels to one-hot encoding

        # computes gradients for each layer
        dz3 = self.a3 - y_one_hot  # gradient for output layer
        dw3 = (self.a2.T @ dz3) / m
        db3 = np.sum(dz3, axis=0, keepdims=True) / m

        dz2 = (dz3 @ self.W3.T) * self.relu_deriv(self.z2)  # gradient for layer 2
        dw2 = (self.a1.T @ dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        dz1 = (dz2 @ self.W2.T) * self.relu_deriv(self.z1)  # gradient for layer 1
        dw1 = (self.x.T @ dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        # applying L1 and L2 regularization
        if self.l1 > 0:
            dw3 += self.l1 * np.sign(self.W3)
            dw2 += self.l1 * np.sign(self.W2)
            dw1 += self.l1 * np.sign(self.W1)

        if self.l2 > 0:
            dw3 += self.l2 * self.W3
            dw2 += self.l2 * self.W2
            dw1 += self.l2 * self.W1

        # updates weights and biases using gradient descent
        self.W3 -= lr * dw3
        self.b3 -= lr * db3
        self.W2 -= lr * dw2
        self.b2 -= lr * db2
        self.W1 -= lr * dw1
        self.b1 -= lr * db1

    @staticmethod
    def relu(x):
        # ReLU activation
        return np.maximum(0, x)

    @staticmethod
    def relu_deriv(x):
        # derivation of ReLU activation for backpropagation
        return (x > 0).astype(float)

    @staticmethod
    def softmax(x):
        # softmax function normalizes outputs to probabilities
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # exponentiates inputs
        return e_x / np.sum(e_x, axis=1, keepdims=True) # normalizes to get probabilities

    @staticmethod
    def one_hot_encode(y, num_classes):
        # converts labels to one-hot encoded format
        return np.eye(num_classes)[y]

    @staticmethod
    def cross_entropy_loss(y, y_hat):
        # computes cross-entropy loss between true labels and predicted probabilities
        m = y.shape[0]
        m = y.shape[0]
        eps = 1e-12
        y_hat_clipped = np.clip(y_hat, eps, 1. - eps)
        log_probs = -np.log(y_hat_clipped[np.arange(m), y])
        return np.mean(log_probs)

    def fit(self, x_train, y_train, x_val, y_val, lr, epochs, batch_size):
        train_losses = []
        val_accuracies = []

        for epoch in range(1, epochs + 1):
            perm = np.random.permutation(x_train.shape[0]) # Shuffle the training data
            x_train_shuffled, y_train_shuffled = x_train[perm], y_train[perm]

            epoch_loss = 0.0
            num_batches = int(np.ceil(x_train.shape[0] / batch_size))

            for i in range(num_batches):
                start = i * batch_size
                end = start + batch_size
                x_batch = x_train_shuffled[start:end] # batch of inputs
                y_batch = y_train_shuffled[start:end] # batch of labels

                # Forward pass, backward pass, and weight update
                self.forward(x_batch)
                self.backward(y_batch, lr)

                epoch_loss += self.cross_entropy_loss(y_batch, self.a3) # updating the epoch loss

            epoch_loss /= num_batches # average loss is defined
            train_losses.append(epoch_loss)

            val_pred = self.predict(x_val)
            val_acc = np.mean(val_pred == y_val)
            val_accuracies.append(val_acc) \

            print(f"Epoch {epoch:02d} | Training Loss: {epoch_loss:.4f} | Value Accuracy: {val_acc:.4f}")

        self.plot_graph(train_losses, val_accuracies)
        return val_accuracies[-1]

    def plot_graph(self, train_losses, val_accuracies):
        if not os.path.exists('results'):
            os.makedirs('results') # creates results director

        fig, ax1 = plt.subplots() # initializes the plot

        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Training Loss', color='tab:blue')
        ax1.plot(range(1, len(train_losses) + 1), train_losses, color='tab:blue', label='Training Loss')
        ax1.tick_params(axis='y', labelcolor='tab:blue') # defines loss subplot

        ax2 = ax1.twinx()
        ax2.set_ylabel('Validation Accuracy', color='tab:orange')
        ax2.plot(range(1, len(val_accuracies) + 1), val_accuracies, color='tab:orange', label='Validation Accuracy')
        ax2.tick_params(axis='y', labelcolor='tab:orange') # defines accuracy subplot

        plt.title('Training Loss and Validation Accuracy over Epochs')

        # dynamically sets the filename
        if self.l1 > 0 and self.l2 > 0:
            result_path = 'results/experiment-3-l1-and-l2.png'
        elif self.l1 > 0:
            result_path = 'results/experiment-3-l1.png'
        elif self.l2 > 0:
            result_path = 'results/experiment-3-l2.png'

        fig.savefig(result_path)
        print(f"Graph saved to: {result_path}")

    def predict(self, x): # predicts class labels for the input data
        probs = self.forward(x)  # forwards pass to get probabilities
        return np.argmax(probs, axis=1)  # returns the class with highest probability

# acquiring the FashionMNIST dataset
train_set = datasets.FashionMNIST(root='.', train=True, download=True)
test_set = datasets.FashionMNIST(root='.', train=False, download=True)

# preprocessing the data by flattening images and normalizing them.
x_train = train_set.data.numpy().reshape(-1, 28 * 28).astype(np.float32) / 255.0
y_train = train_set.targets.numpy()

x_test = test_set.data.numpy().reshape(-1, 28 * 28).astype(np.float32) / 255.0
y_test = test_set.targets.numpy()

# MLP initialization (L1 regularization)
mlp_l1 = MLP(
    input_size=28 * 28,
    hidden_size1=256,
    hidden_size2=256,
    output_size=10,
    weight_scale=1e-2,
    l1 = 1e-6
)

# trains the model
mlp_l1.fit(
    x_train=x_train,
    y_train=y_train,
    x_val=x_test,
    y_val=y_test,
    lr=1e-2,
    epochs=10,
    batch_size=256
)

# tests the model
test_pred_l1 = mlp_l1.predict(x_test)
test_acc_l1 = np.mean(test_pred_l1 == y_test)
print(f"\nFinal test accuracy: {test_acc_l1:.4f}")

# MLP initialization (L2 regularization)
mlp_l2 = MLP(
    input_size=28 * 28,
    hidden_size1=256,
    hidden_size2=256,
    output_size=10,
    weight_scale=1e-2,
    l2 = 1e-4
)

# trains the model
mlp_l2.fit(
    x_train=x_train,
    y_train=y_train,
    x_val=x_test,
    y_val=y_test,
    lr=1e-2,
    epochs=10,
    batch_size=256
)

# tests the model
test_pred_l2 = mlp_l2.predict(x_test)
test_acc_l2 = np.mean(test_pred_l2 == y_test)
print(f"\nFinal test accuracy: {test_acc_l2:.4f}")

# MLP initialization (L1 and L2 regularization)
mlp_l1_l2 = MLP(
    input_size=28 * 28,
    hidden_size1=256,
    hidden_size2=256,
    output_size=10,
    weight_scale=1e-2,
    l1 = 1e-6,
    l2 = 1e-4
)

# trains the model
mlp_l1_l2.fit(
    x_train=x_train,
    y_train=y_train,
    x_val=x_test,
    y_val=y_test,
    lr=1e-2,
    epochs=10,
    batch_size=256
)

# tests the model
test_pred_l1_l2 = mlp_l1_l2.predict(x_test)
test_acc_l1_l2 = np.mean(test_pred_l1_l2 == y_test)
print(f"\nFinal test accuracy: {test_acc_l1_l2:.4f}")