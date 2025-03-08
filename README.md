# Transformer Classifier Fault Injection

This project demonstrates training a transformer-based classifier on the MNIST dataset (or synthetic data if MNIST is unavailable) and evaluating its performance under various fault injection scenarios. The project consists of three main components:

- **Clean Model Training (clean.py):** Trains a transformer classifier without any fault injection.
- **Fault Injection Experiments (fault.py):** Trains a similar model while injecting faults (e.g., dropout, perturbation, bit flip, bias shift) at different severity levels.
- **Visualization (vz.py):** Reads the outputs from the two training runs (saved as `op.txt` and `opfault.txt`) and generates several figures to compare the model performance.

## Project Structure

- **clean.py:**  
  Contains the code to train and evaluate a clean transformer classifier. It loads the MNIST dataset from `mnist.npz` if available; otherwise, it generates synthetic data.

- **fault.py:**  
  Similar to `clean.py`, but with additional functions to inject faults into the model's computations. This script prints the training log and fault injection evaluation results.

- **op.txt:**  
  The standard output log produced when running `clean.py`. You can redirect the console output to this file if needed.

- **opfault.txt:**  
  The standard output log produced when running `fault.py`. Again, redirect the console output to save the log.

- **vz.py:**  
  Contains code to read `op.txt` and `opfault.txt`, parse the training losses and fault injection results, and generate multiple visualizations (saved as PNG figures).

## Requirements

- **Python:** Version 3.7 or above.
- **CUDA:** The code uses [CuPy](https://cupy.dev/) for GPU acceleration. The scripts (both in `clean.py` and `fault.py`) are set to use CUDA Toolkit 12.8. If you are using a different version, update the CUDA path in these scripts accordingly.

## Installation

1. **Clone the repository or download the files.**

2. **Create and activate a virtual environment (recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

