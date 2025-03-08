import os
import cupy as np
import random
import time
import matplotlib.pyplot as plt  # For visualization

# -----------------------------
# GPU Setup (Ensure CUDA Path is Correct)
# -----------------------------
os.environ["CUDA_PATH"] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin")

# -----------------------------
# Setup & Helper Functions
# -----------------------------
np.random.seed(42)
random.seed(42)

def initialize_weights(input_dim, output_dim):
    return np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim)

# -----------------------------
# Fault Injection Function (for later experiments)
# -----------------------------
def inject_faults(layer_output, fault_type="dropout", severity=0.1):
    if fault_type == "dropout":
        mask = np.random.rand(*layer_output.shape) > severity
        return layer_output * mask
    elif fault_type == "perturbation":
        noise = np.random.normal(0, severity, size=layer_output.shape)
        return layer_output + noise
    elif fault_type == "bit_flip":
        flip_mask = (np.random.rand(*layer_output.shape) < severity) * -2 + 1
        return layer_output * flip_mask
    elif fault_type == "bias_shift":
        return layer_output * (1 + severity)
    return layer_output

# -----------------------------
# Gradient Clipping Helper Function
# -----------------------------
def clip_gradients(grad, clip_value=5.0):
    norm = np.linalg.norm(grad)
    if norm > clip_value:
        grad = grad * (clip_value / norm)
    return grad

# -----------------------------
# Linear Layer with Adam Optimizer and Gradient Clipping
# -----------------------------
class Linear:
    def __init__(self, input_dim, output_dim):
        self.W = initialize_weights(input_dim, output_dim)
        self.b = np.zeros((1, output_dim))
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b)
        self.vb = np.zeros_like(self.b)
        self.t = 0
        
    def forward(self, X):
        self.original_shape = X.shape
        X_flat = X.reshape(-1, self.W.shape[0])
        self.X_flat = X_flat
        out_flat = np.dot(X_flat, self.W) + self.b
        return out_flat.reshape(*self.original_shape[:-1], self.W.shape[1])
    
    def backward(self, dout):
        dout_flat = dout.reshape(-1, self.W.shape[1])
        self.dW = np.dot(self.X_flat.T, dout_flat)
        self.db = np.sum(dout_flat, axis=0, keepdims=True)
        dX_flat = np.dot(dout_flat, self.W.T)
        return dX_flat.reshape(*self.original_shape)
    
    def update(self, lr):
        self.dW = clip_gradients(self.dW, clip_value=5.0)
        self.db = clip_gradients(self.db, clip_value=5.0)
        
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-6
        self.t += 1
        self.mW = beta1 * self.mW + (1 - beta1) * self.dW
        self.vW = beta2 * self.vW + (1 - beta2) * (self.dW ** 2)
        mW_corr = self.mW / (1 - beta1 ** self.t)
        vW_corr = self.vW / (1 - beta2 ** self.t)
        self.W -= lr * mW_corr / (np.sqrt(vW_corr) + epsilon)
        
        self.mb = beta1 * self.mb + (1 - beta1) * self.db
        self.vb = beta2 * self.vb + (1 - beta2) * (self.db ** 2)
        mb_corr = self.mb / (1 - beta1 ** self.t)
        vb_corr = self.vb / (1 - beta2 ** self.t)
        self.b -= lr * mb_corr / (np.sqrt(vb_corr) + epsilon)

# -----------------------------
# ReLU Activation Function
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
        Q = self.W_q.forward(X)
        K = self.W_k.forward(X)
        V = self.W_v.forward(X)
        self.scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(X.shape[-1])
        self.attention_weights = np.exp(self.scores) / np.sum(np.exp(self.scores), axis=-1, keepdims=True)
        attn_out = np.matmul(self.attention_weights, V)
        return self.W_o.forward(attn_out)
    
    def backward(self, dout):
        return dout  # Simplified
        
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
        return self.linear2.forward(self.out_relu)
    
    def backward(self, dout):
        d_out_relu = self.linear2.backward(dout)
        d_out1 = self.relu.backward(d_out_relu)
        return self.linear1.backward(d_out1)
    
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
        attn_out = self.self_attn.forward(X)
        self.out1 = X + attn_out  # Residual connection
        ff_out = self.feed_forward.forward(self.out1)
        self.out2 = self.out1 + ff_out  # Residual connection
        return self.out2
    
    def backward(self, dout):
        d_out1 = self.feed_forward.backward(dout)
        d_attn = self.self_attn.backward(dout + d_out1)
        return d_out1 + d_attn
    
    def update(self, lr):
        self.self_attn.update(lr)
        self.feed_forward.update(lr)

# -----------------------------
# Transformer Classifier (Clean Model with Fault Injection Capability)
# -----------------------------
class TransformerClassifierFull:
    def __init__(self, image_size=28, patch_size=7, d_model=128, d_ff=256, num_layers=2, num_classes=10):
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.d_model = d_model
        
        self.patch_embedding = Linear(patch_size * patch_size, d_model)
        self.transformer_layers = [TransformerLayerFull(d_model, d_ff) for _ in range(num_layers)]
        self.classifier = Linear(d_model, num_classes)
    
    def forward(self, X):
        self.batch_size = X.shape[0]
        images = X.reshape(self.batch_size, 28, 28)
        patches = images.reshape(self.batch_size, self.num_patches, -1)
        self.embedded = self.patch_embedding.forward(patches.reshape(-1, patches.shape[-1])).reshape(self.batch_size, self.num_patches, self.d_model)
        out = self.embedded
        for layer in self.transformer_layers:
            out = layer.forward(out)
        self.features_cls = out[:, 0, :]
        return self.classifier.forward(self.features_cls)
    
    def backward(self, dlogits):
        d_features_cls = self.classifier.backward(dlogits)
        d_out = np.zeros((self.batch_size, self.num_patches, self.d_model))
        d_out[:, 0, :] = d_features_cls
        for layer in reversed(self.transformer_layers):
            d_out = layer.backward(d_out)
        d_embedded = d_out
        d_embedded_flat = d_embedded.reshape(-1, self.d_model)
        self.patch_embedding.backward(d_embedded_flat)
    
    def update(self, lr):
        self.patch_embedding.update(lr)
        for layer in self.transformer_layers:
            layer.update(lr)
        self.classifier.update(lr)
    
    def faulty_forward(self, X, fault_type="dropout", severity=0.1):
        self.batch_size = X.shape[0]
        images = X.reshape(self.batch_size, 28, 28)
        patches = images.reshape(self.batch_size, self.num_patches, -1)
        embedded = self.patch_embedding.forward(patches.reshape(-1, patches.shape[-1])).reshape(self.batch_size, self.num_patches, self.d_model)
        embedded_faulty = inject_faults(embedded, fault_type, severity)
        out = embedded_faulty
        for layer in self.transformer_layers:
            out = layer.forward(out)
            out = inject_faults(out, fault_type, severity)
        features_cls = out[:, 0, :]
        return self.classifier.forward(features_cls)

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
# Data Loading (Using MNIST.npz if Available)
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
# Training and Evaluation Functions
# -----------------------------
def train_model(model, X_train, y_train, epochs=100, lr=0.0001, batch_size=128, lr_decay=0.99):
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

def evaluate_faulty_model(model, X_test, y_test, fault_type="dropout", severity=0.1):
    logits = model.faulty_forward(X_test, fault_type, severity)
    preds = np.argmax(softmax(logits), axis=1)
    acc = np.mean(preds == y_test) * 100
    print(f"Fault Type: {fault_type}, Severity: {severity:.2f}, Test Accuracy: {acc:.2f}%")

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
train_model(model, X_train, y_train, epochs=100, lr=0.0001, batch_size=128, lr_decay=0.99)
end_time = time.time()
print("Training time: {:.2f} seconds".format(end_time - start_time))
evaluate_model(model, X_test, y_test)

print("\nEvaluating Fault Injection Experiments:")
fault_types = ["dropout", "perturbation", "bit_flip", "bias_shift"]
severities = [0.05, 0.10, 0.20, 0.30]
results = {fault: [] for fault in fault_types}
for fault in fault_types:
    for severity in severities:
        logits = model.faulty_forward(X_test, fault, severity)
        preds = np.argmax(softmax(logits), axis=1)
        acc = np.mean(preds == y_test) * 100
        results[fault].append(acc)
        print(f"Fault Type: {fault}, Severity: {severity:.2f}, Test Accuracy: {acc:.2f}%")

# Visualization of Fault Injection Results
plt.figure(figsize=(8, 6))
for fault in fault_types:
    plt.plot(severities, results[fault], marker='o', label=fault)
plt.xlabel("Fault Severity")
plt.ylabel("Test Accuracy (%)")
plt.title("Fault Injection: Test Accuracy vs. Fault Severity")
plt.legend()
plt.grid(True)
plt.show()
