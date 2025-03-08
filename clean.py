import os
os.environ["CUDA_PATH"] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin")
    

import cupy as np
import random
import time

# -----------------------------
# Setup & Helper Functions
# -----------------------------
np.random.seed(42)
random.seed(42)

def initialize_weights(input_dim, output_dim):
    return np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim)

# -----------------------------
# Linear Layer with Adam Optimizer (supports batched inputs)
# -----------------------------
class Linear:
    def __init__(self, input_dim, output_dim):
        self.W = initialize_weights(input_dim, output_dim)
        self.b = np.zeros((1, output_dim))
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        # Adam moments and timestep
        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b)
        self.vb = np.zeros_like(self.b)
        self.t = 0
        
    def forward(self, X):
        self.original_shape = X.shape
        X_flat = X.reshape(-1, self.W.shape[0])
        self.X_flat = X_flat  # Cache for backprop
        out_flat = X_flat.dot(self.W) + self.b
        out = out_flat.reshape(*self.original_shape[:-1], self.W.shape[1])
        return out
    
    def backward(self, dout):
        dout_flat = dout.reshape(-1, self.W.shape[1])
        self.dW = self.X_flat.T.dot(dout_flat)
        self.db = np.sum(dout_flat, axis=0, keepdims=True)
        dX_flat = dout_flat.dot(self.W.T)
        dX = dX_flat.reshape(*self.original_shape)
        return dX
    
    def update(self, lr):
        # Adam hyperparameters
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        self.t += 1
        # Update weights
        self.mW = beta1 * self.mW + (1 - beta1) * self.dW
        self.vW = beta2 * self.vW + (1 - beta2) * (self.dW ** 2)
        mW_corr = self.mW / (1 - beta1 ** self.t)
        vW_corr = self.vW / (1 - beta2 ** self.t)
        self.W -= lr * mW_corr / (np.sqrt(vW_corr) + epsilon)
        # Update biases
        self.mb = beta1 * self.mb + (1 - beta1) * self.db
        self.vb = beta2 * self.vb + (1 - beta2) * (self.db ** 2)
        mb_corr = self.mb / (1 - beta1 ** self.t)
        vb_corr = self.vb / (1 - beta2 ** self.t)
        self.b -= lr * mb_corr / (np.sqrt(vb_corr) + epsilon)

# -----------------------------
# ReLU Activation Layer
# -----------------------------
class ReLU:
    def forward(self, X):
        self.X = X
        return np.maximum(0, X)
    def backward(self, dout):
        dX = dout.copy()
        dX[self.X <= 0] = 0
        return dX

# -----------------------------
# Self-Attention Layer (Single Head)
# -----------------------------
class SelfAttention:
    def __init__(self, d_model):
        self.d_model = d_model
        self.W_q = Linear(d_model, d_model)
        self.W_k = Linear(d_model, d_model)
        self.W_v = Linear(d_model, d_model)
        self.W_o = Linear(d_model, d_model)
    def forward(self, X):
        self.X = X  # (batch, seq_len, d_model)
        self.Q = self.W_q.forward(X)
        self.K = self.W_k.forward(X)
        self.V = self.W_v.forward(X)
        d_k = self.d_model
        self.scores = np.matmul(self.Q, self.K.transpose(0,2,1)) / np.sqrt(d_k)
        exp_scores = np.exp(self.scores - np.max(self.scores, axis=-1, keepdims=True))
        self.attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        self.attention_output = np.matmul(self.attention_weights, self.V)
        self.out = self.W_o.forward(self.attention_output)
        return self.out
    def backward(self, dout):
        d_attention_output = self.W_o.backward(dout)
        dV = np.matmul(self.attention_weights.transpose(0,2,1), d_attention_output)
        d_attention_weights = np.matmul(d_attention_output, self.V.transpose(0,2,1))
        dscores = self.attention_weights * (d_attention_weights - 
                    np.sum(d_attention_weights * self.attention_weights, axis=-1, keepdims=True))
        dscores /= np.sqrt(self.d_model)
        dQ = np.matmul(dscores, self.K)
        dK = np.matmul(dscores.transpose(0,2,1), self.Q)
        dX_q = self.W_q.backward(dQ)
        dX_k = self.W_k.backward(dK)
        dX_v = self.W_v.backward(dV)
        dX = dX_q + dX_k + dX_v
        return dX
    def update(self, lr):
        self.W_q.update(lr)
        self.W_k.update(lr)
        self.W_v.update(lr)
        self.W_o.update(lr)

# -----------------------------
# FeedForward Network
# -----------------------------
class FeedForward:
    def __init__(self, d_model, d_ff):
        self.linear1 = Linear(d_model, d_ff)
        self.relu = ReLU()
        self.linear2 = Linear(d_ff, d_model)
    def forward(self, X):
        self.out1 = self.linear1.forward(X)
        self.out_relu = self.relu.forward(self.out1)
        self.out2 = self.linear2.forward(self.out_relu)
        return self.out2
    def backward(self, dout):
        d_out_relu = self.linear2.backward(dout)
        d_out1 = self.relu.backward(d_out_relu)
        dX = self.linear1.backward(d_out1)
        return dX
    def update(self, lr):
        self.linear1.update(lr)
        self.linear2.update(lr)

# -----------------------------
# Transformer Layer (Full Backprop)
# -----------------------------
class TransformerLayerFull:
    def __init__(self, d_model, d_ff):
        self.self_attn = SelfAttention(d_model)
        self.feed_forward = FeedForward(d_model, d_ff)
    def forward(self, X):
        self.X = X
        self.attn_out = self.self_attn.forward(X)
        self.out1 = X + self.attn_out  # Residual connection
        self.ff_out = self.feed_forward.forward(self.out1)
        self.out2 = self.out1 + self.ff_out  # Residual connection
        return self.out2
    def backward(self, dout):
        d_out1 = dout.copy()  # from output residual
        d_ff = self.feed_forward.backward(dout)
        d_out1 += d_ff
        d_attn = self.self_attn.backward(d_out1)
        dX = d_out1 + d_attn
        return dX
    def update(self, lr):
        self.self_attn.update(lr)
        self.feed_forward.update(lr)

# -----------------------------
# Transformer Classifier (Full Model)
# -----------------------------
class TransformerClassifierFull:
    def __init__(self, image_size=28, patch_size=7, d_model=128, d_ff=256, num_layers=2, num_classes=10):
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size)**2
        self.d_model = d_model
        # Patch embedding: a linear layer mapping each patch to d_model
        self.patch_embedding = Linear(patch_size*patch_size, d_model)
        # Create transformer layers
        self.transformer_layers = [TransformerLayerFull(d_model, d_ff) for _ in range(num_layers)]
        # Classifier head: linear layer mapping first patch (CLS token) to class logits
        self.classifier = Linear(d_model, num_classes)
    def forward(self, X):
        # X: (batch, 28*28)
        self.batch_size = X.shape[0]
        images = X.reshape(self.batch_size, 28, 28)
        patches = images.reshape(self.batch_size, self.num_patches, -1)  # (batch, num_patches, patch_dim)
        self.embedded = self.patch_embedding.forward(patches.reshape(-1, patches.shape[-1])).reshape(self.batch_size, self.num_patches, self.d_model)
        out = self.embedded
        for layer in self.transformer_layers:
            out = layer.forward(out)
        self.features_cls = out[:, 0, :]  # Use first patch as CLS token
        self.logits = self.classifier.forward(self.features_cls)
        return self.logits
    def backward(self, dlogits):
        d_features_cls = self.classifier.backward(dlogits)  # (batch, d_model)
        d_out = np.zeros((self.batch_size, self.num_patches, self.d_model))
        d_out[:, 0, :] = d_features_cls
        for layer in reversed(self.transformer_layers):
            d_out = layer.backward(d_out)
        d_embedded = d_out  # (batch, num_patches, d_model)
        d_embedded_flat = d_embedded.reshape(-1, self.d_model)
        self.patch_embedding.backward(d_embedded_flat)
    def update(self, lr):
        self.patch_embedding.update(lr)
        for layer in self.transformer_layers:
            layer.update(lr)
        self.classifier.update(lr)

# -----------------------------
# Loss & Utility Functions
# -----------------------------
def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

def cross_entropy_loss(probs, y):
    m = y.shape[0]
    log_likelihood = -np.log(probs[np.arange(m), y] + 1e-9)
    loss = np.sum(log_likelihood) / m
    grad = probs.copy()
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(m), y] = 1
    grad = (grad - one_hot) / m
    return loss, grad

# -----------------------------
# Training and Evaluation Functions
# -----------------------------
def train_model(model, X_train, y_train, epochs=100, lr=0.001, batch_size=128, lr_decay=0.99):
    num_samples = X_train.shape[0]
    current_lr = lr
    for epoch in range(epochs):
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        X_train = X_train[indices]
        y_train = y_train[indices]
        epoch_loss = 0.0
        for i in range(0, num_samples, batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]
            logits = model.forward(X_batch)
            probs = softmax(logits)
            loss, dloss = cross_entropy_loss(probs, y_batch)
            epoch_loss += loss * X_batch.shape[0]
            model.backward(dloss)
            model.update(current_lr)
        epoch_loss /= num_samples
        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")
        current_lr *= lr_decay
def evaluate_model(model, X_test, y_test):
    logits = model.forward(X_test)
    preds = np.argmax(softmax(logits), axis=1)
    acc = np.mean(preds == y_test) * 100
    print(f"Test Accuracy: {acc:.2f}%")

# -----------------------------
# Data Loading
# -----------------------------
def load_mnist_npz(file_path='mnist.npz'):
    data = np.load(file_path)
    X_train = data['x_train'].astype(np.float32) / 255.0
    X_test = data['x_test'].astype(np.float32) / 255.0
    X_train = X_train.reshape(-1, 28*28)
    X_test = X_test.reshape(-1, 28*28)
    return X_train, data['y_train'], X_test, data['y_test']

def generate_synthetic_mnist(samples=1000, image_size=28):
    X = np.random.rand(samples, image_size*image_size).astype(np.float32)
    y = np.random.randint(0, 10, samples)
    return X, y

# -----------------------------
# Main Execution
# -----------------------------
try:
    X_train, y_train, X_test, y_test = load_mnist_npz('mnist.npz')
except Exception as e:
    print("mnist.npz not found, using synthetic data")
    X_train, y_train = generate_synthetic_mnist(1000)
    X_test, y_test = generate_synthetic_mnist(200)

model = TransformerClassifierFull()
start_time = time.time()
train_model(model, X_train, y_train, epochs=100, lr=0.001, batch_size=128, lr_decay=0.99)
end_time = time.time()
print("Training time: {:.2f} seconds".format(end_time - start_time))
evaluate_model(model, X_test, y_test)
