import re
import matplotlib.pyplot as plt
import numpy as np  # using numpy for visualization (matplotlib requires numpy)

# -----------------------------
# Parse Clean Output (op.txt)
# -----------------------------
with open("op.txt", "r") as f:
    clean_text = f.read()

# Extract epoch loss lines, e.g., "Epoch 1, Loss: 1.0600"
epoch_losses = []
for line in clean_text.splitlines():
    match = re.search(r"Epoch (\d+), Loss:\s*([\d.]+)", line)
    if match:
        epoch = int(match.group(1))
        loss = float(match.group(2))
        epoch_losses.append((epoch, loss))
epoch_losses.sort(key=lambda x: x[0])

# Extract clean test accuracy, e.g., "Test Accuracy: 94.55%"
clean_match = re.search(r"Test Accuracy:\s*([\d.]+)%", clean_text)
if clean_match:
    clean_accuracy = float(clean_match.group(1))
    print(f"Clean Test Accuracy: {clean_accuracy}%")
else:
    clean_accuracy = None
    print("Clean accuracy not found in op.txt")

# -----------------------------
# Parse Fault Injection Output (opfault.txt)
# -----------------------------
fault_results = {}  # Format: {fault_type: {severity: accuracy, ...}, ...}
with open("opfault.txt", "r") as f:
    for line in f:
        # Expected format: "Fault Type: dropout, Severity: 0.05, Test Accuracy: 41.02%"
        m = re.search(r"Fault Type:\s*(\w+),\s*Severity:\s*([\d.]+),\s*Test Accuracy:\s*([\d.]+)%", line)
        if m:
            fault_type = m.group(1)
            severity = float(m.group(2))
            accuracy = float(m.group(3))
            fault_results.setdefault(fault_type, {})[severity] = accuracy

# Sort severities for each fault type.
for fault in fault_results:
    fault_results[fault] = dict(sorted(fault_results[fault].items()))

# -----------------------------
# Figure 1: Clean Model Loss vs. Epoch
# -----------------------------
epochs = [e for e, l in epoch_losses]
losses = [l for e, l in epoch_losses]
plt.figure()
plt.plot(epochs, losses, marker='o', color='blue')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Figure 1: Clean Model - Loss vs. Epoch")
plt.grid(True)
plt.savefig("figure1_loss_vs_epoch.png")
plt.close()

# -----------------------------
# Figure 2: Clean Test Accuracy as a Horizontal Line
# -----------------------------
plt.figure()
if clean_accuracy is not None:
    plt.axhline(y=clean_accuracy, color='green', linestyle="--", label=f"Clean Accuracy: {clean_accuracy}%")
plt.xlabel("Arbitrary X")
plt.ylabel("Test Accuracy (%)")
plt.title("Figure 2: Clean Test Accuracy")
plt.legend()
plt.savefig("figure2_clean_accuracy.png")
plt.close()

# -----------------------------
# Figure 3: Subplots for Each Fault Type (2x2 Grid)
# -----------------------------
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
for ax, (fault_type, data) in zip(axs.flat, fault_results.items()):
    sev = sorted(data.keys())
    acc = [data[s] for s in sev]
    ax.plot(sev, acc, marker='o')
    ax.set_title(f"{fault_type}")
    ax.set_xlabel("Fault Severity")
    ax.set_ylabel("Test Accuracy (%)")
    ax.grid(True)
plt.tight_layout()
plt.savefig("figure3_fault_injection_subplots.png")
plt.close()

# -----------------------------
# Figure 4: Average Fault Injection Accuracy per Fault Type (Bar Plot)
# -----------------------------
avg_accuracies = {}
for fault, data in fault_results.items():
    avg_accuracies[fault] = sum(data.values()) / len(data)
plt.figure()
plt.bar(avg_accuracies.keys(), avg_accuracies.values(), color='orange')
plt.xlabel("Fault Type")
plt.ylabel("Average Test Accuracy (%)")
plt.title("Figure 4: Average Fault Injection Accuracy per Fault Type")
plt.savefig("figure4_avg_accuracy_fault_type.png")
plt.close()

# -----------------------------
# Figure 5: Scatter Plot of Fault Injection Results
# -----------------------------
plt.figure()
for fault_type, data in fault_results.items():
    sev = list(data.keys())
    acc = list(data.values())
    plt.scatter(sev, acc, label=fault_type)
plt.xlabel("Fault Severity")
plt.ylabel("Test Accuracy (%)")
plt.title("Figure 5: Scatter Plot of Fault Injection Results")
plt.legend()
plt.grid(True)
plt.savefig("figure5_scatter_fault_injection.png")
plt.close()

# -----------------------------
# Figure 6: Boxplot of Fault Injection Accuracies per Fault Type
# -----------------------------
plt.figure(figsize=(8, 6))
data_to_plot = [list(data.values()) for fault, data in fault_results.items()]
plt.boxplot(data_to_plot, labels=fault_results.keys())
plt.xlabel("Fault Type")
plt.ylabel("Test Accuracy (%)")
plt.title("Figure 6: Boxplot of Fault Injection Accuracies per Fault Type")
plt.savefig("figure6_boxplot_fault_injection.png")
plt.close()

# -----------------------------
# Figures 7-10: Individual Bar Plots for Each Fault Severity
# -----------------------------
unique_severities = sorted({s for data in fault_results.values() for s in data.keys()})
# For each unique severity, produce a bar plot.
for severity in unique_severities:
    plt.figure()
    fault_types_list = list(fault_results.keys())
    acc_values = [fault_results[f].get(severity, None) for f in fault_types_list]
    plt.bar(fault_types_list, acc_values, color='violet')
    plt.xlabel("Fault Type")
    plt.ylabel("Test Accuracy (%)")
    plt.title(f"Figure for Severity {severity:.2f}: Fault Injection Accuracy")
    plt.savefig(f"figure_bar_severity_{int(severity*100)}.png")
    plt.close()

# -----------------------------
# Figure 11: Heatmap of Fault Injection Accuracies
# (Rows: Fault Types, Columns: Severities)
# -----------------------------
fault_types_list = list(fault_results.keys())
heatmap_data = []
for fault in fault_types_list:
    row = []
    for sev in unique_severities:
        row.append(fault_results[fault].get(sev, np.nan))
    heatmap_data.append(row)
heatmap_data = np.array(heatmap_data)
plt.figure(figsize=(8, 6))
plt.imshow(heatmap_data, cmap='viridis', aspect='auto')
plt.colorbar(label="Test Accuracy (%)")
plt.xticks(range(len(unique_severities)), [f"{sev:.2f}" for sev in unique_severities])
plt.yticks(range(len(fault_types_list)), fault_types_list)
plt.xlabel("Fault Severity")
plt.ylabel("Fault Type")
plt.title("Figure 11: Heatmap of Fault Injection Accuracies")
plt.savefig("figure11_heatmap_fault_injection.png")
plt.close()

# -----------------------------
# Figure 12: Grouped Bar Chart of Absolute Accuracy Drop (Clean - Fault)
# -----------------------------
plt.figure(figsize=(10, 6))
width = 0.2
x = np.arange(len(fault_types_list))
for i, sev in enumerate(unique_severities):
    drops = []
    for fault in fault_types_list:
        fault_acc = fault_results[fault].get(sev, None)
        drop = clean_accuracy - fault_acc if clean_accuracy is not None and fault_acc is not None else 0
        drops.append(drop)
    plt.bar(x + i * width, drops, width, label=f"Severity {sev:.2f}")
plt.xlabel("Fault Type")
plt.ylabel("Accuracy Drop (Clean - Fault) (%)")
plt.title("Figure 12: Absolute Accuracy Drop by Fault Type and Severity")
plt.xticks(x + width * (len(unique_severities) - 1) / 2, fault_types_list)
plt.legend()
plt.savefig("figure12_grouped_bar_accuracy_drop.png")
plt.close()

print("All 12 figures saved.")
