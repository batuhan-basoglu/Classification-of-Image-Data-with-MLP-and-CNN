import numpy as np
from torchvision import datasets

class MLP:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, weight_scale):
        self.W1 = np.random.randn(input_size, hidden_size1) * weight_scale
        self.b1 = np.zeros((1, hidden_size1))
        self.W2 = np.random.randn(hidden_size1, hidden_size2) * weight_scale
        self.b2 = np.zeros((1, hidden_size2))
        self.W3 = np.random.randn(hidden_size2, output_size) * weight_scale
        self.b3 = np.zeros((1, output_size))

    def forward(self, x):
        self.x = x       
        self.z1 = x @ self.W1 + self.b1
        self.a1 = self.relu(self.z1)   
        self.z2 = self.a1 @ self.W2 + self.b2  
        self.a2 = self.relu(self.z2)          
        self.z3 = self.a2 @ self.W3 + self.b3  
        self.a3 = self.softmax(self.z3)    
        return self.a3

    def backward(self, y, lr):
        m = y.shape[0]
        y_one_hot = self.one_hot_encode(y, self.W3.shape[1])

        dz3 = self.a3 - y_one_hot      
        dw3 = (self.a2.T @ dz3) / m                
        db3 = np.sum(dz3, axis=0, keepdims=True) / m

        dz2 = (dz3 @ self.W3.T) * self.relu_deriv(self.z2)
        dw2 = (self.a1.T @ dz2) / m                
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        dz1 = (dz2 @ self.W2.T) * self.relu_deriv(self.z1)
        dw1 = (self.x.T @ dz1) / m                 
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        self.W3 -= lr * dw3
        self.b3 -= lr * db3
        self.W2 -= lr * dw2
        self.b2 -= lr * db2
        self.W1 -= lr * dw1
        self.b1 -= lr * db1

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_deriv(x):
        return (x > 0).astype(float)

    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    @staticmethod
    def one_hot_encode(y, num_classes):
        return np.eye(num_classes)[y]

    @staticmethod
    def cross_entropy_loss(y, y_hat):
        m = y.shape[0]
        eps = 1e-12
        y_hat_clipped = np.clip(y_hat, eps, 1. - eps)
        log_probs = -np.log(y_hat_clipped[np.arange(m), y])
        return np.mean(log_probs)

    def train_model(self, x_train, y_train, x_val, y_val, lr, epochs, batch_size):
        for epoch in range(1, epochs + 1):
            perm = np.random.permutation(x_train.shape[0])
            x_train_shuffled, y_train_shuffled = x_train[perm], y_train[perm]

            epoch_loss = 0.0
            num_batches = int(np.ceil(x_train.shape[0] / batch_size))

            for i in range(num_batches):
                start = i * batch_size
                end   = start + batch_size
                x_batch = x_train_shuffled[start:end]
                y_batch = y_train_shuffled[start:end]

                self.forward(x_batch)
                self.backward(y_batch, lr)

                epoch_loss += self.cross_entropy_loss(y_batch, self.a3)

            epoch_loss /= num_batches

            val_pred = self.predict(x_val)
            val_acc  = np.mean(val_pred == y_val)

            print(f"Epoch {epoch:02d} | Training Loss: {epoch_loss:.4f} | Value Accuracy: {val_acc:.4f}")

        return val_acc

    def predict(self, x):
        probs = self.forward(x)
        return np.argmax(probs, axis=1)


train_set = datasets.FashionMNIST(root='.', train=True, download=True)
test_set  = datasets.FashionMNIST(root='.', train=False, download=True)

x_train = train_set.data.numpy().reshape(-1, 28 * 28).astype(np.float32) / 255.0
y_train = train_set.targets.numpy()

x_test  = test_set.data.numpy().reshape(-1, 28 * 28).astype(np.float32) / 255.0
y_test  = test_set.targets.numpy()

mlp = MLP(
    input_size  = 28 * 28,
    hidden_size1= 128,
    hidden_size2= 64,
    output_size = 10,
    weight_scale= 1e-2
)

mlp.train_model(
    x_train  = x_train,
    y_train  = y_train,
    x_val    = x_test,
    y_val    = y_test,
    lr       = 1e-2,
    epochs   = 10,
    batch_size=128
)

test_pred = mlp.predict(x_test)
test_acc  = np.mean(test_pred == y_test)
print(f"\nFinal test accuracy: {test_acc:.4f}")
